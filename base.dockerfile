FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# Install apt-managed software (version blinded - assuming future versions won't break APIs upon rebuild)
RUN export DEBIAN_FRONTEND=noninteractive \
	&& apt update \
	&& apt install --assume-yes \
    git \
    hisat2 \
    samtools \
    bedtools \
    bedops \
    fastqc \
    stringtie \
    cufflinks \
    wget \
    bc \
    curl \
    default-jre \
    unzip \
    locales \
    r-base \
    r-base-dev \
    bioperl \
    cpanminus \
    r-cran-randomforest \
    kallisto \
    pip \
    ncbi-entrez-direct

# Fixing perl locale setting error
RUN echo 'LANGUAGE=en_US.UTF-8' >> /etc/default/locale \
    && echo 'LC_ALL=en_US.UTF-8' >> /etc/default/locale \
    && echo 'LANG=en_US.UTF-8' >> /etc/default/locale \
    && echo 'LANGUAGE=en_US.UTF-8' >> /etc/environment \
    && echo 'LC_ALL=en_US.UTF-8' >> /etc/environment \
    && echo 'LANG=en_US.UTF-8' >> /etc/environment \
    && locale-gen en_US.UTF-8

# Install PIP-managed software
RUN pip3 install \
    CPAT==3.0.4 \
    jupyter==1.0.0 \
    matplotlib==3.4.3 \
    matplotlib-inline==0.1.3 \
    numpy==1.19.5 \
    pandas==1.3.2 \
    scikit-learn==0.24.2 \
    scikit-plot==0.3.7 \
    scipy==1.7.1 \
    seaborn==0.11.2 \
    shap==0.39.0

# Install R software
RUN R -e "install.packages('ROCR',dependencies=TRUE)" \
    && R -e "install.packages('BiocManager',dependencies=TRUE)" \
    && R -e "BiocManager::install('devtools',dependencies=TRUE,update=TRUE,ask=FALSE)" \
    && R -e "BiocManager::install('remote',dependencies=TRUE,update=TRUE,ask=FALSE)" \
    && R -e "BiocManager::install('pachterlab/sleuth',dependencies=TRUE,update=TRUE,ask=FALSE)"

# Install required cpanm modules
RUN cpanm Parallel::ForkManager@1.07 \
    && cpanm Bio::DB::SeqFeature::Store@1.7.4

# Install UCSC required binaries
RUN cd /usr/bin \
    && wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigBedSummary \
    && wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigSummary \
    && wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bigWigAverageOverBed \
    && chmod 755 bigBedSummary \
    && chmod 755 bigWigSummary \
    && chmod 755 bigWigAverageOverBed

# Install sra-toolkit 
RUN cd /usr/bin \
    && wget --output-document sratoolkit.tar.gz https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.11.3/sratoolkit.2.11.3-ubuntu64.tar.gz \
    && tar -vxzf sratoolkit.tar.gz

# Install fasta_ushuffle from github (FEELnc dependency)
RUN cd /usr/bin \
    && git clone git://github.com/agordon/fasta_ushuffle.git \
    && cd fasta_ushuffle \
    && make

# Install FEELnc from github
RUN cd /usr/bin \
    && git clone https://github.com/tderrien/FEELnc.git

# Setup PATH and other ENV_VARS
ENV FEELNCPATH="/usr/bin/FEELnc"

ENV PERL5LIB="$PERL5LIB:${FEELNCPATH}/lib/"

ENV PATH="/usr/bin/sratoolkit.2.11.3-ubuntu64/bin:/usr/bin/fasta_ushuffle:${FEELNCPATH}/bin/LINUX/:${FEELNCPATH}/scripts/:${FEELNCPATH}/utils/:/usr/bin/${PATH}"

# Fix interactive config of SRAtoolkit (https://github.com/ncbi/sra-tools/issues/409)
RUN echo $("Aexyo" | vdb-config -i)

CMD ["/bin/bash"]
