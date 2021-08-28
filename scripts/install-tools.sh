#!/bin/bash

apt-get update && export DEBIAN_FRONTEND=noninteractive \
&& apt install --assume-yes git ncbi-entrez-direct hisat2 samtools stringtie cufflinks wget curl pip default-jre unzip locales r-base r-base-dev bioperl cpanminus r-cran-randomforest

# Install FastQC
curl -o /usr/bin/fastqc.zip https://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.11.9.zip
unzip -d /usr/bin /usr/bin/fastqc.zip
chmod 755 /usr/bin/FastQC/fastqc

# Install FEELnc
cd /bin
git clone https://github.com/tderrien/FEELnc.git

# Install Fasta-ushuffle
cd /bin
git clone git://github.com/agordon/fasta_ushuffle.git
cd fasta_ushuffle
make
cp fasta_ushuffle ushuffle /bin

# Fixing perl locale setting error
echo 'LANGUAGE=en_US.UTF-8' >> /etc/default/locale
echo 'LC_ALL=en_US.UTF-8' >> /etc/default/locale
echo 'LANG=en_US.UTF-8' >> /etc/default/locale

echo 'LANGUAGE=en_US.UTF-8' >> /etc/environment
echo 'LC_ALL=en_US.UTF-8' >> /etc/environment
echo 'LANG=en_US.UTF-8' >> /etc/environment

locale-gen en_US.UTF-8

# Installing required python packages
pip3 install pandas numpy matplotlib seaborn CPAT
