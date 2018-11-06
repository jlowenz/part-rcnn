#!/usr/bin/env bash

export PYTHONPATH=/code/pkgs/partnet/src:/code/pkgs/part-rcnn:$PYTHONPATH
python3 /code/pkgs/part-rcnn/chairs.py $@
