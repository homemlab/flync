FROM continuumio/anaconda3:latest

SHELL ["/bin/bash", "-c"]

COPY . /bin/app

RUN ["bash", "-c", "conda-env"]

# Preapre app exec
RUN chmod 755 /bin/app/flync

WORKDIR /bin/app

CMD ["python3", "flync"]
