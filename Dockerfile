FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

COPY . /bin/app

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && . /bin/app/scripts/install-tools.sh \
    && . /bin/app/scripts/install-sratoolkit.sh \
    && chmod 755 /bin/app/dlinct

RUN ["cpanm", "https://cpan.metacpan.org/authors/id/S/SZ/SZABGAB/Parallel-ForkManager-1.07.tar.gz"]

RUN ["cpanm", "Bio::DB::SeqFeature::Store"]

ENV FEELNCPATH="/bin/FEELnc"

ENV PERL5LIB="$PERL5LIB:${FEELNCPATH}/lib/"

ENV PATH="${FEELNCPATH}/bin/LINUX/:${FEELNCPATH}/scripts/:${FEELNCPATH}/utils/:/usr/bin/FastQC:/bin/app:${PATH}"

CMD ["/bin/bash"]
