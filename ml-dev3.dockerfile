FROM rfcdsantos/dlinct:base

RUN git clone https://github.com/rf-santos/dlinct.git -b ml-dev && \
mv /dlinct/ /bin/app

# Preapre app exec
RUN chmod 755 /bin/app/dlinct

WORKDIR /bin/app

CMD ["dlinct"]
