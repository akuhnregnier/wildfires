#!/usr/bin/env sh
#PBS -N copernicus
#PBS -l select=1:ncpus=1:mem=7gb
#PBS -l walltime=0:32:00
#PBS -J 0-31

# 148 slices to cover all the months
# ie. -J 0-147

/rds/general/user/ahk114/home/.pyenv/versions/miniconda3-latest/bin/python3 /rds/general/user/ahk114/home/Documents/wildfires/data/cx1_scripts/copernicus_analysis.py
