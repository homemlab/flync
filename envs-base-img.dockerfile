FROM continuumio/miniconda3:latest

ENV TERM=xterm-256color

SHELL ["/bin/bash"; "-c"]

RUN mkdir /bin/app

COPY ./conda-env /bin/app

COPY ./env /bin/app/env

RUN /bin/bash -c chmod 755 /bin/app/conda-env \
    && bash /bin/app/conda-env

RUN conda clean -a -y
