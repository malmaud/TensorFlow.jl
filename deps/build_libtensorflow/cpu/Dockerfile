FROM tensorflow/tensorflow:1.1.0-rc0-devel
ADD upstream_patch /tensorflow
ADD build_libtf.sh /tensorflow
WORKDIR /tensorflow
RUN git apply upstream_patch
VOLUME /out
CMD ["/bin/bash", "/tensorflow/build_libtf.sh"]
