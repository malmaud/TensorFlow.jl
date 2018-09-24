FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update -y && apt-get install -y make gcc g++ bzip2 hdf5-tools unzip gfortran curl
WORKDIR /
RUN mkdir -p /opt/julia-1.0.0 && \
    curl -s -L https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.0-linux-x86_64.tar.gz | tar -C /opt/julia-1.0.0 -x -z --strip-components=1 -f -
ENV PYTHON /usr/bin/python
ENV JUPYTER /usr/local/bin/jupyter
ENV TF_USE_GPU 1
ADD setup.jl .
RUN /opt/julia-1.0.0/bin/julia setup.jl
RUN echo "\nPATH=/opt/julia-1.0.0/bin:\$PATH\n" >> /root/.bashrc
EXPOSE 8888
CMD ["/opt/julia-1.0.0/bin/julia"]
