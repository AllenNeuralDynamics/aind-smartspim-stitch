# 200 seconds aprox. to build the image

FROM ubuntu:18.04
FROM openjdk
FROM maven:3.8.6
COPY . /home
RUN apt-get update -y
RUN chmod +x /home/installing_terastitcher_ubuntu18.04.sh
ADD TeraStitcher-portable-1.11.10-with-BF-Linux.tar.gz /home/
RUN ./home/installing_terastitcher_ubuntu18.04.sh
RUN pip install --upgrade autopep8
RUN autopep8 -i /home/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py
ENV PATH "$PATH:/home/TeraStitcher-portable-1.11.10-with-BF-Linux"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/java/openjdk/lib/server"
CMD ["/bin/bash"]