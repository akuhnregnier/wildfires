#!/usr/bin/env sh
#PBS -N chelsa
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=0:35:00
#PBS -J 0-1679

# 1680 files

/rds/general/user/ahk114/home/.pyenv/versions/miniconda3-latest/bin/python3 /rds/general/user/ahk114/home/Documents/cx1_scripts/chelsa_analysis.py
