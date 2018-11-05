#!/usr/bin/env bash

export PYTHONPATH=/code/pkgs/partnet/src:/code/pkgs/part_rcnn:$PYTHONPATH
python3 /code/pkgs/part_rcnn/chairs.py $@
