FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenmpi-dev \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN cd /home && git clone --single-branch --branch fix/data_paths https://github.com/camilolaiton/TeraStitcher.git && \
    mkdir build-terastitcher && \
    cd build-terastitcher && \
    cmake -DWITH_UTILITY_MODULE_teraconverter=ON -DWITH_UTILITY_MODULE_mdatagenerator=ON -DWITH_UTILITY_MODULE_volumeconverter=ON -DWITH_UTILITY_MODULE_mergedisplacements=ON ../TeraStitcher/src && \
    make -j `nproc` && \
    # need ownership of /usr/local to install without sudo
    # chown -R ${USER}:${USER} /usr/local/ && \
    make install && \
    pip install --upgrade autopep8 && \
    autopep8 -i ../TeraStitcher/src/utils/pyscripts/*.py

# ENV PATH "$PATH:/data/TeraStitcher-portable-1.11.10-with-BF-Linux"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/java/openjdk/lib/server"

# Java setup
# OpenJDK setup
RUN cd /home && \ 
    wget https://cdn.azul.com/zulu/bin/zulu8.62.0.19-ca-jdk8.0.332-linux_x64.tar.gz && \
    tar -xzf zulu8.62.0.19-ca-jdk8.0.332-linux_x64.tar.gz
ENV JAVA_HOME "/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/jre/lib/amd64/server"

# Maven setup (for n5-spark utilities)
RUN cd /home && \ 
    wget https://dlcdn.apache.org/maven/maven-3/3.9.1/binaries/apache-maven-3.9.1-bin.tar.gz && \
    tar -xzf apache-maven-3.9.1-bin.tar.gz
ENV PATH "/home/apache-maven-3.9.1/bin:$PATH"
ENV PATH "/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/bin:$PATH"

RUN echo "localhost slots=70" > /home/hostfile

# exiftool to validate image metadata
RUN cd /home && \
    wget https://exiftool.org/Image-ExifTool-12.57.tar.gz --no-check-certificate && \
    gzip -dc Image-ExifTool-12.57.tar.gz | tar -xf -

ENV PATH "/home/Image-ExifTool-12.57:$PATH" 

# Creating a new conda environment for argschema python bug
RUN conda create -n smart_stitcher python=3.8

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "smart_stitcher", "/bin/bash", "-c"]
RUN conda install -c conda-forge awscli

RUN pip install -U --no-cache-dir \
    'tifffile < 2022.4.22' \
    xarray_multiscale==1.1.0 \
    argschema==3.0.4 \
    mpi4py \
    bokeh==2.4.2 \
    zarr==2.13.3 \
    pint==0.20.1 \
    aind-data-schema==0.13.52 \
    xmltodict==0.13.0 \
    natsort==8.2.0 \
    pims==0.6.1 \
    dask[distributed]==2022.11.1 \
    PyExifTool==0.5.5 \
    tqdm==4.64.1 \
    git+https://github.com/AllenNeuralDynamics/aind-ng-link.git \
    git+https://github.com/AllenNeuralDynamics/pystripe.git@feature/output_format \
    git+https://github.com/AllenNeuralDynamics/ome-zarr-py.git@feature/delayed-dask-poc \
    aicsimageio@git+https://github.com/camilolaiton/aicsimageio.git@feature/zarrwriter-multiscales-daskjobs

# ENV MALLOC_TRIM_THRESHOLD_ 0