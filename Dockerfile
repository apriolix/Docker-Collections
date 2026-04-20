# nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda@sha256:ef33852f3d321c9aedee5103f57b247114407d2e8382fe291a7ea5b2e6cb94ce AS base

ENV DEBIAN_FRONTEND=noninteractive

#----------------------------------------
# 2. System-dependencies
#----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
  && add-apt-repository universe \
  && add-apt-repository multiverse \
  && apt-get update  \
  && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config unzip yasm git checkinstall \
      libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
      python3-dev python3-pip python3-numpy \
      libjpeg-dev libpng-dev libtiff-dev \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      libxvidcore-dev libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
      libopencore-amrnb-dev libopencore-amrwb-dev \
      libdc1394-dev \
      libv4l-dev v4l-utils \
      libtbb2 libtbb-dev libatlas-base-dev gfortran \
      libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
      libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
  && rm -rf /var/lib/apt/lists/*

#----------------------------------------
# Opencl / not fully working yet
#----------------------------------------
# OpenCL & Blas/FFT
RUN apt-get update && apt-get install -y --no-install-recommends \
      ocl-icd-opencl-dev \
      opencl-headers \
      ocl-icd-libopencl1 \
      clinfo \
      nvidia-opencl-dev \
      libclblas-dev \
      libclfft-dev \
      pocl-opencl-icd \
  && rm -rf /var/lib/apt/lists/*




#----------------------------------------
# 3. Removing of already installed packages
#----------------------------------------
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 uninstall -y opencv-python opencv-python-headless || true && \
    apt-get remove -y python3-opencv || true && \
    rm -rf /usr/local/lib/python3*/dist-packages/cv2*

# OpenCv requires numpy 1
RUN pip3 install "numpy<2" 


#----------------------------------------
# 4. OpenCV-Source code cloning
#----------------------------------------
ARG OPENCV_VERSION=4.12.0
WORKDIR /opt
RUN git clone --depth 1  \
      https://github.com/opencv/opencv.git && \
    git clone --depth 1  \
      https://github.com/opencv/opencv_contrib.git

#----------------------------------------
# 5. Init build directory
#----------------------------------------
RUN mkdir -p /opt/opencv_build
WORKDIR /opt/opencv_build


ARG CUDA_ARCH_BIN=6.1

RUN cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_python_bindings_generator=ON \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_PC_FILE_NAME=opencv.pc \
  -D PYTHON3_EXECUTABLE="$(which python3)" \
  -D PYTHON3_INCLUDE_DIR="$(python3 -c 'from sysconfig import get_paths; print(get_paths()[\"include\"])')" \
  -D PYTHON3_NUMPY_INCLUDE_DIRS="$(python3 -c 'import numpy; print(numpy.get_include())')" \
  # CUDA
  -D WITH_CUDA=ON \
  -D CUDA_TOOLKIT_ROOT_DIR="$(dirname "$(dirname "$(which nvcc)")")" \
  -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
  -D CUDA_ARCH_PTX="${CUDA_ARCH_BIN}" \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D WITH_CUBLAS=ON \
  -D WITH_CUFFT=ON \
  -D WITH_NVCUVID=ON \
  -D BUILD_opencv_cudaarithm=ON \
  -D BUILD_opencv_cudaimgproc=ON \
  -D BUILD_opencv_cudafilters=ON \
  -D BUILD_opencv_cudaoptflow=ON \
  -D BUILD_opencv_cudawarping=ON \
  -D BUILD_opencv_cudaobjdetect=ON \
  -D BUILD_opencv_cudastereo=ON \
  -D BUILD_opencv_cudalegacy=ON \
  -D BUILD_opencv_cudabgsegm=ON \
  -D BUILD_opencv_cudacodec=ON \
  # DNN with CUDA-acceleration
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_CUDNN=ON \
  # OpenCL (T-API / UMat)
  -D WITH_OPENCL=ON \
  -D BUILD_opencv_opencl=ON \
  -D OPENCV_OPENCL_RUNTIME=dynamic \
  -D WITH_OPENCL_SVM=ON \
  -D WITH_OPENCLAMDBLAS=ON \
  -D WITH_OPENCLAMDFFT=ON \
  # Additional features
  -D WITH_V4L=ON \
  -D WITH_OPENGL=ON \
  -D WITH_GSTREAMER=ON \
  -D WITH_TBB=ON \
  -D BUILD_opencv_world=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_DOCS=OFF \
  ../opencv



#----------------------------------------
# 6. compilation and istallation of opencv
#----------------------------------------
RUN make -j"$(nproc)" install && \
    ldconfig


FROM base AS python_dnn

RUN python3 -m pip install torch tensorflow scikit-learn keras 
RUN python3 -m pip install ultralytics --no-deps
RUN python3 -m pip install torchvision
RUN python3 -m pip install cupy

RUN pip3 install "numpy<2" 


FROM python_dnn AS ros

RUN apt update && apt install locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8

SHELL ["/bin/bash", "-c"]

# Build-Arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble

ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

RUN apt install locales && locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 
ENV LANG=en_US.UTF-8

# Installation of dependencies
RUN add-apt-repository universe && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    lsb-release \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-argcomplete \
    libeigen3-dev \
    libboost-all-dev \
    libssl-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libpng-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Adding ROS 2 repository and key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-desktop \
    && rm -rf /var/lib/apt/lists/*

# Setting up ROS2 in /etc/profile.d for global use (global)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/profile.d/ros.sh 


# Setting up ROS_FAST_DDS
COPY ./fast_dds_ip4.xml /etc/ros/fast_dds_ip4.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/etc/ros/fast_dds_ip4.xml


FROM ros AS webots 
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}
RUN apt install locales && locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 
ENV LANG=en_US.UTF-8

ARG WEBOTS_VERSION=2023.1.0

ENV WEBOTS_VERSION=${WEBOTS_VERSION}
ENV PATH="/opt/webots/bin:$PATH"

# System-Update und Basis-Tools (kein empfohlenes Zusatz-Overhead)
RUN apt install software-properties-common && add-apt-repository universe && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    lsb-release \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-argcomplete \
    libeigen3-dev \
    libboost-all-dev \
    libssl-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libpng-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    software-properties-common \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 repository and key adding
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor --batch --yes \
    > /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*




# Install required python packages
RUN pip3 install --no-cache-dir scipy matplotlib pybind11 eigenpy

# Webots downloading and installation
RUN mkdir -p /etc/apt/keyrings && cd /etc/apt/keyrings && wget -q https://cyberbotics.com/Cyberbotics.asc \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/Cyberbotics.asc] https://cyberbotics.com/debian binary-amd64/" | tee /etc/apt/sources.list.d/Cyberbotics.list \
    && apt update && apt install webots -y

# Setting up environement variables /etc/profile.d setzen (global)
RUN echo "export ROS_DOMAIN_ID=0" > /etc/profile.d/ros.sh \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/profile.d/ros.sh \
    && echo "export PATH=/opt/webots/bin:\$PATH" >> /etc/profile.d/webots.sh \
    && echo "export LD_LIBRARY_PATH=/opt/webots/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/webots.sh

# Workspace and build (ROS 2)
RUN useradd -ms /bin/bash robot \
    && mkdir -p /home/robot/robot_ws/src \
    && chown -R robot:robot /home/robot



# Final building
RUN /bin/bash -lc "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install || true"

RUN python3 -m pip install "numpy<2"
RUN python3 -m pip install numba 
RUN python3 -m pip install ImageHash 


RUN python3 -m pip install tensorflow-keras


FROM webots AS additional_cpp

# Debugging with cmake
RUN apt-get update && apt-get install -y gdb

# FROM ros AS realsense

# RUN python3 -m pip install pyrealsense2

# ## c++ sdk for ros packages https://github.com/realsenseai/librealsense/blob/master/doc/installation.md
# RUN apt-get update && apt-get install -y libssl-dev  \
#     libusb-1.0-0-dev \
#     libudev-dev \ 
#     pkg-config \ 
#     libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at

# WORKDIR /tmp
# RUN git clone https://github.com/realsenseai/librealsense.git

# # Permission scripts
# WORKDIR /tmp/librealsense
# RUN sed -i 's/sudo //g' ./scripts/setup_udev_rules.sh \
#     && chmod +x ./scripts/*.sh \
#     && ./scripts/setup_udev_rules.sh || true

# # SDK building
# RUN mkdir build && cd build \
#     && cmake ../ -DBUILD_CV_EXAMPLES=true && make -j"$(nproc)" install


# FROM realsense AS realsense_ros

# ARG REALSENSE_ROS_WS=realsense_ros_ws
# WORKDIR /${REALSENSE_ROS_WS}/src
# RUN git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master 
# WORKDIR /${REALSENSE_ROS_WS}

# RUN apt install -y python3-colcon-core python3-colcon-common-extensions


# RUN apt-get install python3-rosdep -y
# RUN rosdep init && rosdep update \
#     && rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y 
    
# RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build

# RUN echo "source /${REALSENSE_ROS_WS}/install/local_setup.bash" >> /etc/profile.d/ros.sh   

# ENV REALSENSE_ROS_WS=${REALSENSE_ROS_WS}


# WORKDIR /WallBuildBot
