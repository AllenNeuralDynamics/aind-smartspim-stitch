FROM mfisherman/openmpi AS stage1

FROM python:3.8 AS stage2
COPY --from=stage1 /project /home/
LABEL org.opencontainers.image.authors="camilo.laiton@alleninstitute.org"

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    "build-essential" \
    "libopenmpi-dev" \
    "make" \
    "software-properties-common" \
    "git" && \
    pip install --upgrade pip && \
    pip install --upgrade autopep8

# Installing pystripe and terastitcher with fixed bugs
RUN git clone https://github.com/chunglabmit/pystripe.git home/pystripe && \
    cd /home/pystripe && python setup.py install && \
    curl 'https://unicampus365-my.sharepoint.com/personal/g_iannello_unicampus_it/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fg%5Fiannello%5Funicampus%5Fit%2FDocuments%2FTeraStitcher%5Fdistributions%2F1%2E11%2F1%2E11%2E10%2FTeraStitcher%2Dportable%2D1%2E11%2E10%2Dwith%2DBF%2DLinux%2Etar%2Egz' \
    -H 'authority: unicampus365-my.sharepoint.com' \
    -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
    -H 'accept-language: en-US,en;q=0.9' \
    -H 'cookie: rtFa=ByhNcjshNe371Bzr23X9aaQDEEQMfo3ZavURF6PXFfcmMzI2NjlDRDYtNzM3Ri00QjM5LThCREQtRDY5NTExMjBEM0ZDIzEzMzA1MjIzOTMyMjk4OTI2NSNCREU5NUJBMC1EMDkwLUQwMDAtNTU4Qy05NjUwMjZCOTk3MzEjQ0FNSUxPLkxBSVRPTiU0MEFMTEVOSU5TVElUVVRFLk9SRxA7VAnrHMGueWYkrkJZ4GYqHTsapmdl4Yp7W4YjZIIP/nkba/m31jiy9uP+CaNyyrFwtGgYzpSuS7lZB1850uDVAdhvIW7ee5iiF/aZNxqQ5ZR58OxAMOwqUeU3UzDzzaRbUHWevM1mHXoKBvaqbEgmhfRybd8W26wzMkSL1r1MbSrQr8qR7kHh68+yufHOnh+C8hcQCZegoUZ4ATvbz7zRNxdGOdlAz5OQ+/7E2fav5A953ZzBVvsvRaPRE0QwajOsCD3plX1aqoibrJK2twKuGGV1x5qu4XOb98RBF1VvqVcFrggl8Zpch3IuMBNhbWFKcKjIG0bea79MI5mEFGygAAAA; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEyLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uIzgwMDc4NGRiMmM3NjE4OWQ1Nzg3OWU1NjVlYjhkMGI0MDNhMTE1YWY4ZDMzMTk3Y2Q1OWE2YzY2MDAwZjY5MDEsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jODAwNzg0ZGIyYzc2MTg5ZDU3ODc5ZTU2NWViOGQwYjQwM2ExMTVhZjhkMzMxOTdjZDU5YTZjNjYwMDBmNjkwMSwxMzMwODE2MzU1ODAwMDAwMDAsMCwxMzMwODI0OTY1ODUxNDU2MDAsMC4wLjAuMCwyNTgsZDM2M2UwNzMtMjFkZi00ZGY1LWE4OWYtOWViZGI3MmI1NmFiLCwsZTZkYzY2YTAtNjA3Zi01MDAwLTA0NWQtYzExYzU5NTE2ZmY3LGU2ZGM2NmEwLTYwN2YtNTAwMC0wNDVkLWMxMWM1OTUxNmZmNyxQS3AreVdzcWUwcUpWVDJrT3FlNGVnLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCxwK3c1R1JKRFdUeTZuclppY3QvY2h2VG9xekdRZHR3a3FLT2ZwS255eDlwQ2tWaFBVbEJIendhZjRNclF0UVFsMTdSblUwa1ArRmpVUzArWHUwRWlDY2NsT2lqRzdWbEFnZkNLZjhIQXBBOXdETjBrd2h1QXZudkg3eUo2UjEvaEtEMW9KUVR5UzFJTWh4dlNEMHhNam40N1lrNElSMnQ3M3YzeUdFSzcwNXBMaXNkNzd0Q2pwRi9EV21YN3ZJU0gvcngrU3N3a3lxaGxEUkd4K2ZnSlRDTXNGakNOMW14NlFFNGxGR0VablNlWnpjbEVJR0FVNm41MUNUZEcrTmRuL2RpMlpUTGJFR0x6TXdyUTBZK0YrUFhua2xzUWp3c1VBckwyRGp0cWE1dWFLK1ZEYktPSFFVcERSVU9pdWYzWWlWMnR4aDV0OVhWbnl1SkY4aXozbUE9PTwvU1A+; KillSwitchOverrides_enableKillSwitches=; KillSwitchOverrides_disableKillSwitches=' \
    -H 'if-none-match: "{858EA2E3-2FB0-41DC-85EA-068EDCB8294E},2"' \
    -H 'referer: https://unicampus365-my.sharepoint.com/personal/g_iannello_unicampus_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fg%5Fiannello%5Funicampus%5Fit%2FDocuments%2FTeraStitcher%5Fdistributions%2F1%2E11%2F1%2E11%2E10%2FTeraStitcher%2Dportable%2D1%2E11%2E10%2Dwith%2DBF%2DLinux%2Etar%2Egz&parent=%2Fpersonal%2Fg%5Fiannello%5Funicampus%5Fit%2FDocuments%2FTeraStitcher%5Fdistributions%2F1%2E11%2F1%2E11%2E10&ga=1' \
    -H 'sec-ch-ua: "Google Chrome";v="105", "Not)A;Brand";v="8", "Chromium";v="105"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-platform: "Windows"' \
    -H 'sec-fetch-dest: iframe' \
    -H 'sec-fetch-mode: navigate' \
    -H 'sec-fetch-site: same-origin' \
    -H 'service-worker-navigation-preload: true' \
    -H 'upgrade-insecure-requests: 1' \
    -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36' \
    --compressed \
    --output /home/TeraStitcher-portable-1.11.10-with-BF-Linux.tar.gz

# Installing terastitcher
RUN cd /home && tar -xzf TeraStitcher-portable-1.11.10-with-BF-Linux.tar.gz && \
    cd /home/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts && rm Parastitcher.py && \
    wget https://raw.githubusercontent.com/camilolaiton/TeraStitcher/fix/data_paths/src/utils/pyscripts/Parastitcher.py && \
    wget https://raw.githubusercontent.com/camilolaiton/TeraStitcher/fix/data_paths/src/utils/pyscripts/paraconverter.py && \
    autopep8 -i /home/TeraStitcher-portable-1.11.10-with-BF-Linux/pyscripts/*.py

ENV PATH "$PATH:/home/TeraStitcher-portable-1.11.10-with-BF-Linux"
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

CMD ["/bin/bash"]