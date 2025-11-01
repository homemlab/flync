FROM rfcdsantos/flync:base-envs

COPY . /bin/app

# Install the Python package
RUN cd /bin/app && pip install -e .

WORKDIR /bin/app

CMD ["flync", "--help"]
