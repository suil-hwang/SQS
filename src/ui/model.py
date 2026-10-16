# src/ui/model.py
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, fields, replace

import numpy as np

from geometry.direction_augmentation import augment_directions
from geometry.vector import normalize_vectors
from sqs.dual import DualGraphData, DualVertex
from sqs.primal import PrimalGraphData
from sqs.structure import SQS
from ui.state import VolumeState, LayerVisibility, ValidationState
from ui.sqs_solver import SolverConfig


@dataclass(frozen=True, slots=True, eq=False)
class PreparedSolveDirectionSet:
    """Effective solve directions and the state needed to restore the input set."""

    direction_vectors: np.ndarray
    input_direction_vectors: np.ndarray | None = None
    auxiliary_direction_count: int = 0

    @property
    def has_auxiliary_directions(self) -> bool:
        return self.auxiliary_direction_count > 0

    def is_realized_by(self, dual: DualGraphData) -> bool:
        """Whether a solved dual graph has exactly these directions as vertices, up to a few ulps."""
        positions = dual.vertex_positions
        return positions.shape == self.direction_vectors.shape and np.allclose(
            positions, self.direction_vectors, rtol=0.0, atol=8.0 * np.finfo(np.float64).eps,
        )


@dataclass(frozen=True, slots=True)
class GraphCounts:
    """Arithmetic counts of the displayed graphs; they are not an SQS audit."""

    dual_vertices: int
    dual_edges: int  # Counted with multiplicity.
    dual_faces: int
    single_support_edges: int
    double_support_edges: int
    digon_faces: int
    dual_face_sizes: tuple[tuple[int, int], ...]  # (face size, face count), ascending.
    primal_vertices: int
    primal_edges: int
    primal_faces: int

    @classmethod
    def from_graphs(cls, dual: DualGraphData, primal: PrimalGraphData) -> GraphCounts:
        """Count the elements of the dual graph Q* and the primal graph Q."""
        sizes = Counter(len(face.vertex_indices) for face in dual.faces)
        return cls(
            dual_vertices=dual.num_directions, dual_edges=dual.num_edges_with_multiplicity,
            dual_faces=dual.num_faces, single_support_edges=dual.num_single_support_edges,
            double_support_edges=dual.num_double_support_edges, digon_faces=sum(face.is_digon for face in dual.faces),
            dual_face_sizes=tuple(sorted(sizes.items())), primal_vertices=primal.num_vertices,
            primal_edges=primal.num_edges, primal_faces=primal.num_faces,
        )

    @property
    def euler_characteristic(self) -> int:
        """Euler characteristic V - E + F of the primal graph Q."""
        return self.primal_vertices - self.primal_edges + self.primal_faces

    def summary(self) -> str:
        """Return a console line checking E*, F* against (2n, n+2) and, if Q has edges, V, E, F against (n+2, 2n, n)."""
        n = self.dual_vertices
        dual_ok = (self.dual_edges, self.dual_faces) == (2 * n, n + 2)
        text = (
            f"n={n}: dual V*={n}, E*={self.dual_edges}, F*={self.dual_faces} {'[OK]' if dual_ok else '[!]'}"
            f" ({self.single_support_edges} single, {self.double_support_edges} double pairs;"
            f" {self.digon_faces} digon, {self.dual_faces - self.digon_faces} regular faces)"
        )
        if self.primal_edges:
            primal_ok = (self.primal_vertices, self.primal_edges, self.primal_faces) == (n + 2, 2 * n, n)
            text += (
                f"; primal V={self.primal_vertices}, E={self.primal_edges}, F={self.primal_faces},"
                f" Euler {self.euler_characteristic} {'[OK]' if primal_ok else '[!]'}"
            )
        return text


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Session snapshot restored on transaction rollback."""

    dual: DualGraphData
    primal: PrimalGraphData
    face_direction_indices: tuple[int, ...] | None
    layer_visibility: LayerVisibility
    validation_state: ValidationState
    volume_state: VolumeState
    solver_config: SolverConfig
    configured_auxiliary_direction_count: int
    prepared_directions: PreparedSolveDirectionSet | None


class SQSModel:
    """Editor session: immutable graph references, face-direction mapping, direction preparation, mutable UI state."""

    def __init__(self) -> None:
        """Create a session with no directions."""
        self.layer_visibility = LayerVisibility()
        self.validation_state = ValidationState()
        self.solver_config = SolverConfig()
        self.volume_state = VolumeState()
        self.configured_auxiliary_direction_count: int = 0
        self.prepared_directions: PreparedSolveDirectionSet | None = None
        self.activate_effective_direction_set(np.empty((0, 3), dtype=np.float64), prepared_directions=None)

    @property
    def sqs(self) -> SQS:
        """Unvalidated SQS container with primal faces in input-direction order; a non-permutation mapping raises."""
        primal, mapping = self.primal, self.face_direction_indices
        if mapping is not None:
            if (
                any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in mapping)
                or sorted(mapping) != list(range(primal.num_faces))
            ):
                raise ValueError("face_direction_indices must be a permutation of the face indices")
            if mapping != tuple(range(primal.num_faces)):
                primal = replace(primal, faces=tuple(face for _, face in sorted(zip(mapping, primal.faces))))
        return SQS(self.dual, primal)

    def invalidate_validation(self) -> None:
        """Discard cached validation; a fresh state is stale until refreshed."""
        self.validation_state = ValidationState()

    def replace_input_direction_set(self, direction_vectors: np.ndarray) -> None:
        """Replace the input directions; reset graphs, volumes, validation, auxiliary count and visibility toggles."""
        self.activate_effective_direction_set(direction_vectors, prepared_directions=None)
        self.configured_auxiliary_direction_count = 0
        self.layer_visibility = LayerVisibility()

    def resolve_input_direction_set(self) -> np.ndarray:
        """Return a copy of the input directions: the stored base set while auxiliary directions are active."""
        prepared = self.prepared_directions
        if prepared is not None and prepared.has_auxiliary_directions and prepared.input_direction_vectors is not None:
            return np.array(prepared.input_direction_vectors, copy=True)
        return self.dual.vertex_positions.copy()

    def prepare_solve_directions(self) -> PreparedSolveDirectionSet:
        """Return effective directions (input plus auxiliary) for the next solve; the session is unchanged."""
        vectors = self.resolve_input_direction_set()
        count = max(0, self.configured_auxiliary_direction_count)
        if count == 0:
            return PreparedSolveDirectionSet(vectors)

        return PreparedSolveDirectionSet(
            direction_vectors=augment_directions(vectors, count), input_direction_vectors=vectors,
            auxiliary_direction_count=count,
        )

    def activate_effective_direction_set(
        self, direction_vectors: np.ndarray, *,
        prepared_directions: PreparedSolveDirectionSet | None,
    ) -> None:
        """Set Q* to the normalized directions without edges, empty Q, and clear mapping, shell and validation.

        `prepared_directions` is kept only while it has auxiliary directions.
        """
        vectors = normalize_vectors(direction_vectors, name="direction_vectors")
        dual = DualGraphData(tuple(DualVertex(direction) for direction in vectors), (), ())
        primal = PrimalGraphData((), (), ())
        self.dual, self.primal = dual, primal
        self.face_direction_indices = None
        self.prepared_directions = (
            prepared_directions
            if prepared_directions is not None and prepared_directions.has_auxiliary_directions else None
        )
        self.volume_state.set_shell_geometry(None)
        self.invalidate_validation()

    def restore_input_direction_set(self) -> None:
        """Reactivate the input directions alone, dropping auxiliary directions and graphs."""
        self.activate_effective_direction_set(self.resolve_input_direction_set(), prepared_directions=None)

    def input_direction_count(self) -> int:
        """Return the number of input directions, excluding auxiliary ones."""
        prepared = self.prepared_directions
        if prepared is not None and prepared.has_auxiliary_directions and prepared.input_direction_vectors is not None:
            return len(prepared.input_direction_vectors)
        return self.dual.num_directions

    def configured_effective_direction_count(self) -> int:
        """Return the next solve's direction count: input plus configured auxiliary."""
        return self.input_direction_count() + max(0, self.configured_auxiliary_direction_count)

    def create_snapshot(self) -> SessionSnapshot:
        """Capture the session for rollback; mutable state is copied, graphs are shared."""
        return SessionSnapshot(
            dual=self.dual,
            primal=self.primal,
            face_direction_indices=self.face_direction_indices,
            layer_visibility=copy.copy(self.layer_visibility),
            validation_state=copy.deepcopy(self.validation_state),
            volume_state=self.volume_state.clone(),
            solver_config=copy.copy(self.solver_config),
            configured_auxiliary_direction_count=self.configured_auxiliary_direction_count,
            prepared_directions=copy.deepcopy(self.prepared_directions),
        )

    def restore_snapshot(self, snapshot: SessionSnapshot) -> None:
        """Reinstate every snapshot field."""
        for info in fields(snapshot):
            setattr(self, info.name, getattr(snapshot, info.name))
