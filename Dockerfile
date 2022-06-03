FROM rfcdsantos/flync:base-with-envs

COPY . /bin/app

RUN chmod 755 /bin/app/flync

ENV PATH=${PATH}:/bin/app

WORKDIR /bin/app

CMD ["flync", "--help"]
