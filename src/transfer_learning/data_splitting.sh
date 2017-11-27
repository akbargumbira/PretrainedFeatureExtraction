#!/usr/bin/env bash
cd ../../
export PYTHONPATH=${PYTHONPATH}:$(pwd)/src

# Random split
python src/transfer_learning/data_splitting.py -r -s 0
python src/transfer_learning/data_splitting.py -r -s 1
python src/transfer_learning/data_splitting.py -r -s 2

# Animal - transportation split
python src/transfer_learning/data_splitting.py
