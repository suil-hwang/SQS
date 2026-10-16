# pyplantri

Enumerates the duals of simple quadrangulations of the sphere with a dedicated
[plantri](https://users.cecs.anu.edu.au/~bdm/plantri/) 5.5 FILTER, `plantri_sqs`, and stores them in compact graph
caches. For a dual vertex count $n$, each plane-map isomorphism class (reflections identified) is emitted once as a
4-regular, 4-edge-connected, loop-free plane multigraph $G^*$; `SimpleQuadrangulation` reads the primal quadrangulation
$G$ ($n + 2$ vertices) from the same darts.

## Installation

From the SQS root, with a C compiler available:

```bash
python -m pip install -e src/pyplantri
```

## Usage

```python
from pyplantri.plantri import PlantriEnumeration, SimpleQuadrangulation
from pyplantri.plane_graph_cache import open_graph_cache, write_graph_cache

batch = PlantriEnumeration.enumerate(6, max_count=100)  # the 9 duals for n = 6
primal = SimpleQuadrangulation(batch.graphs[0])
assert primal.num_vertices == 8 and primal.dual is batch.graphs[0]

write_graph_cache(batch, "n6.cache", graph_class="quartic_multigraph")
assert open_graph_cache("n6.cache").get_by_graph_id(0).twin == batch.graphs[0].twin
```

| `PrimalMinimumDegree`  | plantri flags | Duals                                      |
| ---------------------- | ------------- | ------------------------------------------ |
| `AT_LEAST_2` (default) | `-q -c2 -m2`  | all; parallel edges allowed                |
| `AT_LEAST_3`           | `-q -c2`      | simple (digon-free) only; none for $n < 6$ |

$n$ ranges over 3–61. `max_count` returns a prefix in plantri's order and does not certify that enumeration is
complete.

## License

[Apache License 2.0](src/LICENSE-2.0.txt). Bundles plantri 5.5 by Gunnar Brinkmann, Brendan McKay, and Heidi
Van den Camp.

## References

- G. Brinkmann, S. Greenberg, C. Greenhill, B. D. McKay, R. Thomas, and P. Wollan. Generation of simple quadrangulations
  of the sphere. _Discrete Mathematics_ 305 (2005), 33–54.
- G. Brinkmann and B. D. McKay. Fast generation of planar graphs. _MATCH Commun. Math. Comput. Chem._ 58 (2007),
  323–357.
