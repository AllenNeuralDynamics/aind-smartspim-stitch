# gcloud builds submit --tag us.gcr.io/neural-dynamics-dev/stitch-images/terastitcher

FROM mfisherman/openmpi AS stage1

FROM python:3.8 AS stage2
COPY --from=stage1 /project /home/
LABEL org.opencontainers.image.authors="camilo.laiton@alleninstitute.org"

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    "build-essential" \
    "libopenmpi-dev" \
    "make" \
    "cmake" \
    "software-properties-common" \
    "git" && \
    pip install --upgrade pip && \
    pip install --upgrade autopep8 && \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb http://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean

# Set fallback mount directory
ENV MNT_DIR /mnt/gcs

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
    git clone https://github.com/AllenNeuralDynamics/terastitcher-module home/terastitcher-module && \
    cd /home/terastitcher-module && \
    git checkout remotes/origin/feature/ng_link && \
    pip install -e .

ENTRYPOINT ["python", "/home/terastitcher-module/code/src/terastitcher_module/terastitcher.py", "--pyscripts_path", "/home/TeraStitcher/src/utils/pyscripts", "--align.cpu_params.hostfile", "/home/hostfile", "--merge.cpu_params.hostfile", "/home/hostfile"]
CMD ["--help"]