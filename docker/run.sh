#!/usr/bin/env bash

BASE_CODE=${MRCNN_BASE_CODE:-"INVALID"}
DATA_DIR=${MRCNN_DATA:-"INVALID"}
SDATA_DIR=${MRCNN_SDATA:-"INVALID"}

CMD=${@:-"/bin/bash"}
xhost +
docker run --runtime=nvidia -it --rm \
	      --net=host \
			  --env QT_X11_NO_MITSHM=1 \
	      -p 127.0.0.1:7777:8888 \
        -v $HOME/.pylog.yaml:/home/user/.pylog.yaml \
	      -v $BASE_CODE:/code \
	      -v $DATA_DIR:/data \
				-v $SDATA_DIR:/sdata \
	      -v $HOME/.Xauthority:/home/user/.Xauthority:rw \
	      -v /etc/opt/VirtualGL:/etc/opt/VirtualGL \
	      -e LOCAL_USER_ID=$(id -u) \
	      -e LOCAL_GROUP_ID=$(id -g) \
	      -e DISPLAY=$DISPLAY \
	      -e VGL_CLIENT=$VGL_CLIENT \
	      -e VGL_DISPLAY=:0.0 \
	      --privileged \
	      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	      jlowenz/part-rcnn:py3_tf1.9.0 $CMD
xhost -
