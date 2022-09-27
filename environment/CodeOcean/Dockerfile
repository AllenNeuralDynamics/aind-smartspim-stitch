# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    "build-essential" \
    "libopenmpi-dev" \
    "make" \
    "cmake" \
    "software-properties-common" \
    "git" "wget" && \
    pip install --upgrade pip && \
    pip install --upgrade autopep8

# Installing pystripe and terastitcher with fixed bugs
RUN git clone https://github.com/chunglabmit/pystripe.git home/pystripe && \
    cd /home/pystripe && python setup.py install

RUN cd /home && git clone --single-branch --branch fix/data_paths https://github.com/camilolaiton/TeraStitcher.git && \
    mkdir build-terastitcher && \
    cd build-terastitcher && \
    cmake -DWITH_UTILITY_MODULE_teraconverter=ON -DWITH_UTILITY_MODULE_mdatagenerator=ON -DWITH_UTILITY_MODULE_volumeconverter=ON -DWITH_UTILITY_MODULE_mergedisplacements=ON ../TeraStitcher/src && \
    make -j `nproc` && \
    # need ownership of /usr/local to install without sudo
    # chown -R ${USER}:${USER} /usr/local/ && \
    make install && \
    autopep8 -i ../TeraStitcher/src/utils/pyscripts/*.py

# ENV PATH "$PATH:/home/TeraStitcher-portable-1.11.10-with-BF-Linux"
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
    wget https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz && \
    tar -xzf apache-maven-3.8.6-bin.tar.gz
ENV PATH "/home/apache-maven-3.8.6/bin:$PATH"
ENV PATH "/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/bin:$PATH"

RUN echo "localhost slots=70" > /home/hostfile && \
    git clone --branch feature/ng_link https://github.com/AllenNeuralDynamics/terastitcher-module /root/capsule/code/terastitcher-module && \
    cd /root/capsule/code/terastitcher-module \
    pip install -e .

# CMD ["/bin/bash"]