# Three-dimensional frequency-domain airborne electromagnetic variational inversion

Numerical workflows for three-dimensional resistivity inversion using a learned forward operator and spatially correlated normalizing flows. The three configured cases are OVID, staggered conductors and Orkdalen.

This repository contains the portable source and configuration only. Observations, true models, GN outputs, posterior arrays and forward weights are not included. The original numerical results are retained in a separate local archive.

## Installation

Use Python 3.10. The recorded computation environment used Python 3.10.18, PyTorch 2.8.0, CUDA 12.6 and an NVIDIA RTX 4090D. Install the dependencies in a separate environment:

```text
python -m pip install -r requirements.txt
```

Full inversion requires a CUDA-compatible PyTorch installation and adequate GPU memory. CPU source checks do not load the forward checkpoint or perform inversion.

## Inputs and access

`inputs_to_copy.json` records the expected file destinations, sizes and SHA-256 hashes. Its `source` fields refer to the separately retained project archive; they are relative archive locations, not public download links. The numerical configurations resolve all input paths relative to this directory.

| Directory | Required contents |
| --- | --- |
| `data/ovid/` | Observation table, station indices, true model, GN comparison and archived FD response |
| `data/staggered/` | Observation table, station indices, true model, GN comparison and generation records |
| `data/orkdalen/` | Processed observations, indices, preprocessing manifest, fixed geometry/sign metadata, raw source table and GN comparison |
| `weights/` | Learned forward-operator checkpoint `efdo.pkl` |

For an authorized local copy of the original archive, use:

```text
python copy_inputs.py --source-root "PATH_TO_PROJECT_ARCHIVE" --include-large
```

Alternatively, place supplied files at their `destination` paths in `inputs_to_copy.json`. The exact processed files listed there are required for the frozen configurations; a newly downloaded survey is not automatically equivalent.

Survey information and reports can be consulted through the [NGU report archive](https://openarchive.ngu.no/ngu-xmlui/handle/11250/2639632) and [NGU Geoscience Portal](https://www.ngu.no/en/geological-resources/ngu-geoscience-portal). Requests for relevant survey data and its access conditions should be directed to [NGU](https://www.ngu.no/om-ngu/kontakt-ngu). These resources do not establish that the exact processed Orkdalen table used here is downloadable. Contact the repository maintainers about the synthetic inputs, processing records and forward weights; their availability and permission for transfer must be confirmed separately. The repository does not provide or promise a data/weight download.

Synthetic training reads `observations.txt`. The archived `raw_fd/*.imp` files are separate provenance/comparison responses, not replacements for those training tables. The staggered-conductor observations were generated with the learned forward operator and a recorded noise realization. True models and GN results are used for evaluation, not training initialization or checkpoint selection.

## Check and run

From this directory, source-only checks are:

```text
python verify_package.py
python verify_package.py --imports
```

After the separately supplied inputs are in place:

```text
python verify_package.py --imports --inputs --protocol
python run_case.py --case ovid
python run_case.py --case staggered
python run_case.py --case orkdalen
```

The three `run_case.py` commands above validate source and input files without inversion. Add `--execute` for a full new run, for example:

```text
python run_case.py --case ovid --execute
```

Run one GPU job at a time. Each execution creates a new time-stamped directory under `runs/`; existing output directories are preserved. Source/configuration hashes are checked before execution, and inputs are checked against their recorded hashes. The field workflow binds a new runtime protocol to its current paths and freshly generated point-stage outputs.

## Reproduction scope

Both synthetic workflows start from a uniform 100-ohm-m model, run the recorded point initialization and variational optimization, and save 2,048 full three-dimensional posterior draws. Subsequent evaluation includes resistivity statistics, SSIM, conductor geometry, column conductance and the dependence diagnostic. `--skip-analysis` omits the additional structure/GN analysis.

The field workflow starts from the same uniform half-space, performs the DCT180 point fit and native240 refinement, then runs VI160. It estimates moments from 1,024 full draws, quantiles/events from the first 256 draws and predictive responses from eight draws. The original native continuation requires the selected cold-stage checkpoint to be the identity-coordinate result at step 180; a different selection stops the workflow for inspection.

The portable copy has passed source/configuration integrity checks, CPU imports, input hashing and the initial field protocol check. A complete GPU rerun of this copy has not been performed. New outputs are marked as reproductions; equality across different devices and library versions is not assumed. This entry point does not reproduce every manuscript figure layout or the separate multi-seed and noise experiments. Retained original numerical results remain in the local archive.
