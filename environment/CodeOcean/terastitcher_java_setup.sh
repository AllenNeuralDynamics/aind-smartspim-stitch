#!/usr/bin/env bash

# OpenJDK setup
wget https://cdn.azul.com/zulu/bin/zulu8.62.0.19-ca-jdk8.0.332-linux_x64.tar.gz
tar -xzf zulu8.62.0.19-ca-jdk8.0.332-linux_x64.tar.gz
echo "export JAVA_HOME=$PWD/zulu8.62.0.19-ca-jdk8.0.332-linux_x64" >> ~/.bash_profile

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/jre/lib/amd64/server" >> ~/.bash_profile

# Maven setup (for n5-spark utilities)
wget https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz
tar -xzf apache-maven-3.8.6-bin.tar.gz 
echo "export PATH=$PWD/apache-maven-3.8.6/bin:$PATH" >> ~/.bash_profile

source ~/.bash_profile
echo "export PATH=$PWD/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/bin:$PATH" >> ~/.bash_profile
