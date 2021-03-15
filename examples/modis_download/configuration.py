# -*- coding: utf-8 -*-
"""Configure the downloading and processing of MODIS data.

The location of the different folders is controlled using the following environment
variables:
    - EPHEMERAL: Data output and processing metadata that allow the restarting of the
      downloading and processing scripts
    - TMPDIR: Raw data and temporary files

"""
import os
from pathlib import Path

# Output files and files that persist betweeen runs.
processing_dir = Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006_processing"
output_dir = Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006_0d25"

# Input and temporary data.
data_dir = Path(os.environ["TMPDIR"]) / "MOD15A2Hv006"
temporary_dir = Path(os.environ["TMPDIR"]) / "modis_proc_tmp"
temporary_data_dir = Path(os.environ["TMPDIR"]) / "modis_data_tmp"

processing_dir.mkdir(exist_ok=True)
temporary_dir.mkdir(exist_ok=True)
temporary_data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Environment bin path.
env_bin = Path("~/.pyenv/versions/miniconda3-latest/envs/wildfires/bin").expanduser()

download_file = processing_dir / "downloaded.txt"
download_file.touch()

processed_file = processing_dir / "processed.txt"
processed_file.touch()
