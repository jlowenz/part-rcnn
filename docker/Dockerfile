from chaneyk/tensorflow:v1.6.0-py

# COMPONENT: :mirrors / 1.0
RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial main restricted universe multiverse" > /etc/apt/sources.list && \
	echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
	echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-security main restricted universe multiverse" >> /etc/apt/sources.list && \
	echo "deb mirror://mirrors.ubuntu.com/mirrors.txt xenial-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
	DEBIAN_FRONTEND=noninteractive apt-get update

SHELL ["/bin/bash", "-c"]
# COMPONENT: :cppdev / 1.0
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y upgrade && \
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

# COMPONENT: :scipy / 0.18.1
RUN pip install -U scipy

# COMPONENT: :sklearn / 0.18.1
RUN pip install -U scikit-learn==0.19.1

# COMPONENT: :skimage / 0.12.3
RUN pip install -U scikit-image

# COMPONENT: :keras / latest
RUN pip install -U PyYAML \
	h5py \
	jupyter \
	tensorflow-gpu \
	Keras \
	elephas

# COMPONENT: :java / 8
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common && \
	add-apt-repository -y ppa:webupd8team/java

RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y upgrade && \
	apt-get -y install oracle-java8-installer

RUN update-java-alternatives -s java-8-oracle

ENV JAVA_HOME=/usr/lib/jvm/java-8-oracle

# COMPONENT: :opengl / 1.0
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y upgrade && \
	apt-get -y install xserver-xorg-dev \
	libxt-dev \
	mesa-utils \
	mesa-common-dev \
	mesa-utils-extra \
	libgl1-mesa-dev \
	libglapi-mesa


# COMPONENT: :vtk / 7.1.0
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y upgrade && \
	apt-get -y install doxygen

RUN git clone https://github.com/Kitware/VTK /tmp/vtk7.1.0 && \
	pushd /tmp/vtk7.1.0 && \
	git checkout tags/v7.1.0 && \
	popd && \
	pushd /tmp/vtk7.1.0 && \
	mkdir -p build && \
	pushd build && \
	cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_DOCUMENTATION=ON -DVTK_USE_CXX11_FEATURES=ON -DVTK_WRAP_JAVA=ON -DVTK_WRAP_PYTHON=ON .. && \
	make -j $(nproc) install && \
	popd && \
	popd && \
	rm -rf /tmp/vtk7.1.0

# COMPONENT: :pcl / 1.8.0
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y upgrade && \
	apt-get -y install libeigen3-dev \
  libboost-all-dev \
	libflann-dev \
	libopenni-sensor-primesense0 \
	libopenni-sensor-primesense-dev \
	libopenni-dev \
	libopenni0 \
	openni-utils \
	libqhull-dev \
	libqhull7 \
	libopenni2-dev \
	openni2-utils

RUN git clone http://github.com/PointCloudLibrary/pcl.git /tmp/pcl1.8.1 && \
	pushd /tmp/pcl1.8.1 && \
	git checkout tags/pcl-1.8.1 && \
	popd && \
	pushd /tmp/pcl1.8.1 && \
	mkdir -p build && \
	pushd build && \
	cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
	make -j $(nproc) install && \
	popd && \
	popd && \
	rm -rf /tmp/pcl1.8.1
