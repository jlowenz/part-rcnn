#!/usr/bin/env bash

BASE_CODE=${MRCNN_BASE_CODE:-"INVALID"}
DATA_DIR=${MRCNN_DATA:-"INVALID"}

CMD=${@:-"/bin/bash"}
xhost +
nvidia-docker run --rm -it \
	      --net=host \
	      -p 127.0.0.1:7777:8888 \
        -v $HOME/.pylog.yaml:/home/user/.pylog.yaml \
	      -v $BASE_CODE:/code \
	      -v $DATA_DIR:/data \
	      -v $HOME/.Xauthority:/home/user/.Xauthority:rw \
	      -v /etc/opt/VirtualGL:/etc/opt/VirtualGL \
	      -e LOCAL_USER_ID=$(id -u) \
	      -e LOCAL_GROUP_ID=$(id -g) \
	      -e DISPLAY=unix$DISPLAY \
	      -e VGL_CLIENT=$VGL_CLIENT \
	      -e VGL_DISPLAY=:0.0 \
	      --privileged \
	      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	      jlowenz/part-rcnn:py3_tf1.9.0 $CMD
xhost -
