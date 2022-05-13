FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

COPY . /bin/app

RUN conda install -n base pathlib \
    && chmod 755 /bin/app/conda-env \
    && bash /bin/app/conda-env

# Preapre app exec
RUN chmod 755 /bin/app/flync

ENV PATH=${PATH}:/bin/app

WORKDIR /bin/app

CMD ["python3", "flync", "--help"]
