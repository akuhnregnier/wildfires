#!/usr/bin/env bash

to_install=()
while read package;
do
    # echo "$package"
    if [[ $package != \#* ]]; then
        to_install+=("$package")
    fi
done < requirements.txt

echo "Installing: ${to_install[@]}"
conda install --yes "${to_install[@]}"
