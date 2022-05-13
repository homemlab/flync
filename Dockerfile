FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

COPY . /bin/app

# Required for ballgown to execute
RUN mkdir -p /tmp \
&& cd ~/tmp \
&& wget -c wget http://ftp.debian.org/debian/pool/main/r/readline6/libreadline6_6.3-8+b3_amd64.deb \
&& wget -c http://ftp.debian.org/debian/pool/main/g/glibc/multiarch-support_2.19-18+deb8u10_amd64.deb \
&& sudo apt install ./libreadline6_6.3-8+b3_amd64.deb  ./multiarch-support_2.19-18+deb8u10_amd64.deb \
&& rm -rf /tmp \
&& sudo add-apt-repository universe \
&& sudo apt-get install libncurses5

RUN chmod 755 /bin/app/conda-env \
    && bash /bin/app/conda-env \
    && conda install -n base pathlib -y \

# Preapre app exec
RUN chmod 755 /bin/app/flync

ENV PATH=${PATH}:/bin/app

WORKDIR /bin/app

CMD ["python3", "flync", "--help"]
