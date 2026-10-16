# Construction of Simple Quadrangulations of the Sphere at Skeletal Joints

Suil Hwang and Q Youn Hong · _Computer Graphics Forum_ (Pacific Graphics 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2610.17803-b31b1b.svg)](https://arxiv.org/abs/2610.17803)

Given an unordered set of $n$ direction vectors, this code builds a simple quadrangulation of the sphere for a skeletal
joint. It selects the joint topology from enumerated candidate dual graphs, realizes it on the sphere without crossings
by a conflict-constrained assignment with lazy cyclic-order separation (CP-SAT), recovers the primal quadrangulation,
and converts it into exact rational Bézier shell and column volumes.

## Installation

Requires conda and a C compiler for the bundled plantri (e.g., MSVC Build Tools on Windows).
Tested on Windows 11 with Python 3.12.

```bash
git clone https://github.com/suil-hwang/SQS.git
cd SQS
conda env create -f environment.yml
conda activate SQS
python -m pip install -e src/pyplantri
```

Generate the candidate graph catalogues in `resources/plantri/`:

```bash
python src/plane_graphs_generator.py --n 3-10
python src/plane_graphs_generator.py --n 3-20 --digon-free-only
```

ALL catalogues are supported for $n = 3$–$10$, and DIGON_FREE catalogues for $n = 3$–$20$.
For $n > 10$, use first-convex search over DIGON_FREE; exhaustive ALL search is unsupported.

## Usage

```bash
python src/interactive_editor.py
```

Generate $n = 3$–$20$ directions, click `Auto Build (Dual + Primal)`, then `Generate Shell` and `Generate Columns`.

## Citation

```bibtex
@article{10.1111:cgf.70634,
  journal   = {Computer Graphics Forum},
  title     = {{Construction of Simple Quadrangulations of the Sphere at Skeletal Joints}},
  author    = {Hwang, Suil and Hong, Q Youn},
  year      = {2026},
  publisher = {The Eurographics Association and John Wiley \& Sons Ltd.},
  ISSN      = {1467-8659},
  DOI       = {10.1111/cgf.70634}
}
```

## License

The code is released under the [MIT License](LICENSE). `src/pyplantri` bundles plantri 5.5 by Gunnar Brinkmann,
Brendan McKay, and Heidi Van den Camp; it is distributed under the
[Apache License 2.0](src/pyplantri/src/LICENSE-2.0.txt).
