FROM rfcdsantos/dlinct:base

COPY . /bin/app

# Preapre app exec
RUN chmod 755 /bin/app/dlinct

WORKDIR /bin/app

CMD ["dlinct"]
