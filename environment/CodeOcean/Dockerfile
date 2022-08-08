# 200 seconds aprox. to build the image

# FROM ubuntu:18.04
# FROM continuumio/anaconda3
# RUN conda install numpy==1.14
# FROM openjdk
# FROM maven:3.8.6
# COPY . /home
# RUN apt-get update -y
# RUN apt-get install -y git
# RUN git clone https://github.com/chunglabmit/pystripe.git home/pystripe
# RUN cd /home/pystripe; python setup.py install
# RUN chmod +x /home/installing_terastitcher_ubuntu18.04.sh
# ADD TeraStitcher-portable-1.11.10-with-BF-Linux.tar.gz /home/
# RUN ./home/installing_terastitcher_ubuntu18.04.sh
# RUN pip install --upgrade autopep8
# RUN autopep8 -i /home/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py
# ENV PATH "$PATH:/home/TeraStitcher-portable-1.11.10-with-BF-Linux"
# ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/java/openjdk/lib/server"
# CMD ["/bin/bash"]

FROM ubuntu:18.04
# FROM continuumio/anaconda3
RUN apt-get update -y \
    && apt-get install -y --no-install-recomends \
    "build-essential" \
    "libopenmpi-dev" \
    "make" \
    "software-properties-common" \
    "python3" \
    "python3-pip" \
    "git"

RUN curl -O https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz
RUN git clone https://github.com/chunglabmit/pystripe.git
RUN cd /pystripe; python setup.py install
FROM openjdk
FROM maven:3.8.6
COPY . /home
ADD TeraStitcher-portable-1.11.10-with-BF-Linux.tar.gz /home/
# RUN chmod +x /home/installing_terastitcher_ubuntu18.04.sh
# RUN ./home/installing_terastitcher_ubuntu18.04.sh
# RUN pip install --upgrade autopep8
# RUN autopep8 -i /home/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py
ENV PATH "$PATH:/home/TeraStitcher-portable-1.11.10-with-BF-Linux"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/java/openjdk/lib/server"
CMD ["/bin/bash"]