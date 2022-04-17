FROM continuumio/anaconda3:latest

COPY . /bin/app

RUN chmod 755 /bin/app/conda-env \
    && bash conda-env

# Preapre app exec
RUN chmod 755 /bin/app/flync

WORKDIR /bin/app

CMD ["python3", "flync"]
