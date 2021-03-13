# -*- coding: utf-8 -*-
import os
from pathlib import Path

processing_dir = Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006_processing"
data_dir = Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006"
temporary_dir = Path(os.environ["EPHEMERAL"]) / "modis_proc_tmp"
temporary_data_dir = Path(os.environ["EPHEMERAL"]) / "modis_data_tmp"
output_dir = Path(os.environ["EPHEMERAL"]) / "MOD15A2Hv006_0d25"

processing_dir.mkdir(exist_ok=True)
temporary_dir.mkdir(exist_ok=True)
temporary_data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# Environment bin path.
env_bin = Path("~/.pyenv/versions/miniconda3-latest/envs/" "wildfires/bin").expanduser()

download_file = processing_dir / "downloaded.txt"
download_file.touch()

processed_file = processing_dir / "processed.txt"
processed_file.touch()
