FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y python3.6 python3-pip

RUN ln -s /usr/bin/python3.6 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip


## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt