# hash:sha256:08a5f66ca4a011a8b28edd7d97128b706fa30f850fc07310033071d0feb5b270
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libblosc-dev=1.17.1+ds1-1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=conda-forge --channel=bioconda \
        fiji==20220414 \
        openjdk==8.0.332 \
    && conda clean -ya

RUN conda create -n smartspim_stitch python=3.9 -y
SHELL ["conda", "run", "-n", "smartspim_stitch", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    dask==2024.1.1 \
    dask-image==2023.8.1 \
    numpy==1.26.3 \
    pathlib==1.0.1 \
    psutil==5.9.5 \
    regex==2023.10.3 \
    toml==0.10.2 \
    zarr==2.16.1 \
    natsort==8.4.0 \
    matplotlib==3.9.2 \
    aind-data-schema==1.0.0

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server

SHELL ["/bin/bash", "-c"]

RUN conda create -n bigstitcher_env python=3.9 -y
    
COPY postInstall /
RUN /postInstall
