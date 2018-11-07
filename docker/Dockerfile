from tensorflow/tensorflow:1.9.0-devel-gpu-py3

ENV DEBIAN_FRONTEND noninteractive

# COMPONENT: :mirrors / 1.0
RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial main restricted universe multiverse" > /etc/apt/sources.list && \
	echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
	echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-security main restricted universe multiverse" >> /etc/apt/sources.list && \
	echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
	DEBIAN_FRONTEND=noninteractive apt-get update

SHELL ["/bin/bash", "-c"]

# COMPONENT: :cppdev / 1.0
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y install build-essential \
	cmake \
	curl \
	git \
	hashalot \
	mercurial \
	pkg-config \
	python \
	python-dev \
	wget

# COMPONENT: :openblas / latest
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y install gfortran \
	libatlas-dev \
	libatlas-base-dev \
	libblas-dev \
	libflann-dev \
	libgfortran3 \
	libgfortran-4.8-dev \
	liblapack-dev

RUN echo "/opt/OpenBLAS/lib" > /etc/ld.so.conf.d/openblas.conf

WORKDIR /tmp
RUN git clone https://github.com/xianyi/OpenBLAS /tmp/openblas && \
	pushd /tmp/openblas && \
	LD_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/5:$LD_LIBRARY_PATH make -j $(nproc) NO_AFFINITY=1 USE_OPENMP=1 && \
	make -j $(nproc) install && \
	popd && \
	rm -rf /tmp/openblas

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y install doxygen

LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/lib/nvidia-410:${LD_LIBRARY_PATH}

ENV DEBIAN_FRONTEND noninteractive
RUN LC_ALL=C.UTF-8 add-apt-repository -y ppa:nschloe/cmake-nightly && \
		apt-get update && apt-get -y install cmake

RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libxt-dev
RUN apt-get install -y graphviz vim sudo

RUN cd /tmp && \
    git clone https://github.com/jlowenz/pypcd && cd pypcd && \
    sudo python3 setup.py install
RUN pip3 install joblib tfquaternion numba numpy-quaternion scikit-image \
    vtk pydot PyYAML

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y x-window-system mesa-utils

# COMPONENT: :opengl / 1.0
RUN DEBIAN_FRONTEND=noninteractive \
	apt-get -y install xserver-xorg-dev \
	libxt-dev \
	mesa-utils \
	mesa-common-dev \
	mesa-utils-extra \
	libgl1-mesa-dev \
	libglapi-mesa

RUN pip3 install --upgrade --no-cache-dir comet_ml


RUN apt-get install -y gdb libglfw3-dev

RUN pip3 install plotly
RUN pip3 install Cython
RUN pip3 install glumpy
RUN pip3 install PyWavefront
RUN pip3 install ray

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get -y update && apt-get -y install python3.6

RUN pip3 install --upgrade --no-cache-dir comet_ml
#RUN pip3 install --upgrade --no-cache-dir tensorflow-gpu
RUN pip3 install --upgrade pip
RUN pip3 install toposort networkx
RUN pip3 install opencv-contrib-python

# install VirtualGl 2.5.1
RUN apt-get install -y \
    libglu1-mesa-dev mesa-utils xterm libxv1 kmod && \
    wget http://downloads.sourceforge.net/project/virtualgl/2.5.1/virtualgl_2.5.1_amd64.deb && \
    dpkg -i virtualgl_2.5.1_amd64.deb && \
    /opt/VirtualGL/bin/vglserver_config -config +s +f -t && \
    rm virtualgl_2.5.1_amd64.deb && \
    apt-get remove -y wget && \
    rm -rf /var/lib/apt/lists/*

# Install docker in the image
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - &&\
		add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" &&\
		apt-get update && apt-get install -y docker-ce

RUN apt-get install -y net-tools

RUN pip3 install imgaug
RUN pip3 install Keras

RUN apt-get install -y python3-tk
RUN pip3 install matplotlib

# COMPONENT: :user-entry / 1.0
ADD user_entrypoint.sh /usr/bin/user_entrypoint.sh
ENTRYPOINT ["/usr/bin/user_entrypoint.sh"]
