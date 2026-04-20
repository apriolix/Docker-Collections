# Docker Collections

This repository contains a set of CUDA-based Dockerfiles for robotics, computer vision, and deep learning workflows.

## Included files

- `Dockerfile.cuda`
  - Base image built from `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`
  - Installs shared system dependencies used by derived stages
  - Provides common build tools, Python support, multimedia libraries, and OpenCL components

- `Dockerfile.ros_cuda`
  - Extends the CUDA base image with ROS Humble desktop support
  - Sets up ROS 2 environment and Fast DDS configuration

- `Dockerfile.ros_webots_cuda`
  - Extends the ROS CUDA image with Webots installation
  - Adds Webots-specific environment setup and Python dependencies

- `Dockerfile.opencv_cuda`
  - Extends the CUDA base image for building OpenCV from source
  - Uses base image dependencies for OpenCV build and GPU/OpenCL support

- `Dockerfile.python_dnn`
  - Extends the CUDA base image for Python deep learning workloads
  - Installs Torch, TensorFlow, scikit-learn, Keras, Ultralytics, torchvision, and CuPy

- `Dockerfile.ros_realsense_cuda`
  - ROS + RealSense support on top of the CUDA base image

- `Dockerfile.combined_image`
  - Multi-stage Dockerfile that combines outputs from several derived stages into a single final image

- `docker-compose.docker_in_docker.yaml`
  - Compose configuration for Docker-in-Docker scenarios

- `Dockerfile.docker_in_docker`
  - Dockerfile for running Docker inside Docker

## Purpose

The repository is designed to centralize shared system dependencies in a single base image and keep derived images focused on their specific tooling and runtime requirements.
