FROM rfcdsantos/flync:base-envs

COPY . /bin/app

RUN conda activate mapMod && \
    bash /bin/app/scripts/download_tracks.sh && \
    if [ $TRACKS_DOWNLAOD_STATUS != 1 ]; then echo 'Problem downloading required UCSC tracks. Aborting build.'; exit 1; fi && \
    conda deactivate && \
    conda clean --all -y

RUN chmod 755 /bin/app/flync

ENV PATH=${PATH}:/bin/app

WORKDIR /bin/app

CMD ["flync", "--help"]
