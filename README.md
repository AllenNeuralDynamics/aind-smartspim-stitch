# aind-smartspim-stitch

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Programming Languages](https://img.shields.io/github/languages/count/AllenNeuralDynamics/aind-smartspim-stitch)](https://github.com/AllenNeuralDynamics/aind-smartspim-stitch)
![Code Style](https://img.shields.io/badge/code%20style-black-black)

Tile stitching module for teravoxel-scale SmartSPIM light-sheet microscopy datasets. Wraps two stitching backends.

| Backend | Method | Parallelism |
|---------|--------|-------------|
| **TeraStitcher** | Normalized cross-correlation, XML displacement maps | MPI (`mpirun`) |
| **BigStitcher** | Phase correlation (Spark) -> global optimization | Apache Spark (Java, via `run_classes.sh`) |

Both backends read from Code Ocean's standard `../data/` folder layout and write results to `../results/`.

---

## Repository layout

```
code/
├── aind_smartspim_stitch/
│   ├── algorithms/
│   │   ├── bigstitcher.py          # BigStitcher phase-correlation pipeline
│   │   ├── bigstitcher_xml_builder.py  # BDV/SpimData XML generation
│   │   └── terastitcher.py         # TeraStitcher MPI pipeline + OME-Zarr conversion
│   ├── bigstitcher_spark_scripts/
│   │   └── run_classes.sh          # Spark job launcher (called by bigstitcher.py)
│   ├── params/                     # ArgSchema parameter definitions + default YAML configs
│   ├── utils/utils.py              # Shared utilities
│   ├── validate_datasets.py        # Dataset tile-count validation
│   └── zarr_converter/             # OME-Zarr writer (used by TeraStitcher pipeline)
├── run_bigstitcher.py              # Entry point — BigStitcher, local data
├── run_bigstitcher_co_pipeline.py  # Entry point — BigStitcher, S3 data (CO pipeline)
├── run_terastitcher.py             # Entry point — TeraStitcher
├── scripts/
│   ├── terastitcher_parallel.sh    # MPI helper (Linux)
│   └── terastitcher_parallel.bat   # MPI helper (Windows)
└── tests/
```

---

## Required inputs

All entry-point scripts expect the following files in `../data/`:

| File | Purpose |
|------|---------|
| `processing_manifest.json` | Pipeline parameters (stitching channel, CPU count, …) |
| `data_description.json` | Dataset name and metadata |
| `acquisition.json` | Voxel resolution (used to set stitching scale) |

---

## Installation

### Code Ocean (recommended)

The capsule Docker image installs all dependencies. No manual installation needed.

### Local / development

Requires Python ≥ 3.8. Install the package in editable mode:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-smartspim-stitch
cd aind-smartspim-stitch
pip install -e .[dev]
```

#### TeraStitcher (for the TeraStitcher backend)

Install the [command-line version with BioFormats support](https://github.com/abria/TeraStitcher/wiki/Binary-packages#terastitcher-portable-with-support-for-bioformats-command-line-version) for your platform. Then set `pyscripts_path` in the config to the directory containing `parastitcher.py` and `paraconverter.py`.

Known input-parameter bugs in the TeraStitcher Python scripts are fixed in [this fork](https://github.com/camilolaiton/TeraStitcher/tree/fix/data_paths).

#### BigStitcher (for the BigStitcher backend)

Set the `BIGSTITCHER_HOME` environment variable to the BigStitcher Spark installation directory before running.

---

## Running

### TeraStitcher

```bash
cd code
python run_terastitcher.py
```

Configuration is loaded from `aind_smartspim_stitch/params/default_terastitcher_config.yaml` and overridden by the `processing_manifest.json` stitching section.

Key YAML parameters:

```yaml
import_data:
  ref1: H
  ref2: V
  ref3: D
  vxl1: 1.800   # X resolution (µm) — overridden from acquisition.json
  vxl2: 1.800   # Y resolution (µm)
  vxl3: 2.000   # Z resolution (µm)
  additional_params: [sparse_data, libtiff_uncompress]

align:
  cpu_params:
    number_processes: 16   # overridden from processing_manifest.json
  subvoldim: 100
```

**Note:** The full pipeline (stitching -> OME-Zarr conversion) requires roughly **3× the raw dataset size** in scratch space. The `clean_output` flag (default `true`) removes intermediate files after conversion.

### BigStitcher — standalone (local data)

```bash
cd code
python run_bigstitcher.py
```

Reads preprocessed tiles from `../data/preprocessed_data/<channel>/`, computes phase-correlation transforms via Spark, and writes BigDataViewer XML + processing metadata to `../results/`.

### BigStitcher — Code Ocean pipeline (S3 data)

```bash
cd code
python run_bigstitcher_co_pipeline.py
```

Same as above but the BDV XML image-loader path points to the S3 URI read from `../data/path_to_cloud_*`.

---

## Configuration reference

### `processing_manifest.json` (stitching section)

```json
{
  "pipeline_processing": {
    "stitching": {
      "channel": "Ex_488_Em_525",
      "cpus": 16
    }
  }
}
```

### Default TeraStitcher config

See [`code/aind_smartspim_stitch/params/default_terastitcher_config.yaml`](code/aind_smartspim_stitch/params/default_terastitcher_config.yaml) for the full list of tunable parameters.

---

## Testing

```bash
cd code
coverage run -m unittest discover && coverage report
```

Individual test files:

| File | Covers |
|------|--------|
| `tests/test_utils.py` | Utility functions including `wavelength_to_hex` |
| `tests/test_zarr_converter.py` | OME-Zarr writer, pyramid computation |
| `tests/test_terastitcher.py` | TeraStitcher pipeline helpers |
| `tests/test_bigstitcher.py` | BigStitcher downsample estimation, shift calculation |

---

## Contributing

### Code style

```bash
black .          # auto-format
isort .          # sort imports
flake8 .         # lint
interrogate .    # docstring coverage
```

### Commit messages

Follow [Angular commit style](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit):

```
<type>(<scope>): <short summary>
```

Types: `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`

### Pull requests

Internal contributors: create a branch. External contributors: fork and open a PR.

### Documentation

Regenerate RST sources:
```bash
sphinx-apidoc -o doc_template/source/ code/aind_smartspim_stitch
sphinx-build -b html doc_template/source/ doc_template/build/html
```

---

## Links

- [TeraStitcher](https://github.com/abria/TeraStitcher)
- [BigStitcher](https://imagej.net/plugins/bigstitcher/)
- [TeraStitcher documentation (PDF)](https://unicampus365-my.sharepoint.com/:b:/g/personal/g_iannello_unicampus_it/EYT9KbapjBdGvTAD2_MdbKgB5gY_h9rlvHzqp6mUNqVhIw?e=s8GrFC)
