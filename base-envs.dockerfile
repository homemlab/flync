FROM continuumio/miniconda3:latest

ENV TERM=xterm-256color

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /bin/app

COPY ./conda-env /bin/app

RUN mkdir -p /bin/app/env

COPY ./env /bin/app/env

RUN chmod 755 /bin/app/conda-env \
    && bash /bin/app/conda-env

RUN conda clean -a -y
