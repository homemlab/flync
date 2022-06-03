FROM rfcdsantos/flync:local-tracks

RUN chmod 755 /bin/app/flync

ENV PATH=${PATH}:/bin/app

WORKDIR /bin/app

CMD ["flync", "--help"]
