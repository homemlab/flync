FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

COPY . /bin/app

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && . /bin/app/scripts/install-tools.sh \
    && . /bin/app/scripts/install-sratoolkit-stack.sh \
    && chmod 755 /bin/app/dlinct

ENV PATH="/usr/bin/FastQC:/bin/app:${PATH}"

CMD ["/bin/bash"]
