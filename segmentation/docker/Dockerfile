FROM python:3.8.8-slim

RUN apt-get -y update  && apt-get install -y \
  gcc \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential \
  ffmpeg \
  libsm6 \
  libxext6 \
  awscli \
&& rm -rf /var/lib/apt/lists/*;


RUN apt-get -y update  && apt-get install -y \
  gdal-bin \
  libgdal-dev \
  python3-gdal;

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal/
RUN pip install --upgrade pip

RUN pip install --upgrade setuptools
RUN pip install numpy==1.18.4
RUN pip install gdal==2.4.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"
RUN pip --no-cache-dir install torch==1.7.1
WORKDIR /

COPY requirements.txt /

RUN mkdir -p /root/.aws
COPY .aws/* /root/.aws/

RUN pip install -r requirements.txt


