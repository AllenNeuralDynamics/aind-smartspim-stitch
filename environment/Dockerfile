ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

LABEL maintainer="Camilo Laiton <camilo.laiton@alleninstitute.org>"

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

# Set environment variables
ENV AWS_REGION=us-west-2
ENV JAVA_OPTS="-Djava.awt.headless=true"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libblosc-dev=1.21.1+ds2-2 \
        maven=3.6.3-5 && \
    rm -rf /var/lib/apt/lists/*

# Download and set up Java
WORKDIR /home
RUN wget -q https://cdn.azul.com/zulu/bin/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz && \
    tar -xzf zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz && \
    rm zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz

ENV JAVA_HOME="/home/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64"
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server"

RUN conda install -y --freeze-installed \
        ca-certificates=2024.7.4 \
        openssl=3.3.1 \
    && conda clean -ya

WORKDIR /.code-server
RUN wget -qO code-server.tar.gz https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz && \
    tar -xvf code-server.tar.gz --strip-components=1 && \
    rm code-server.tar.gz && \
    ln -s /.code-server/bin/code-server /usr/bin/code-server

# Install Python aind-data-schema
RUN pip install -U --no-cache-dir aind-data-schema==1.4.0

# Setting BigStitcher Home dir that will be installed in postInstall
ENV BIGSTITCHER_HOME=/home/BigStitcher-Spark

# Copy and run post-install script
COPY postInstall /
RUN chmod +x /postInstall && /bin/bash /postInstall

# Set working directory
WORKDIR /home
