FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update -y && apt-get install -y make gcc g++ bzip2 hdf5-tools unzip gfortran curl
WORKDIR /
RUN mkdir -p /opt/julia-0.5.1 && \
    curl -s -L https://julialang.s3.amazonaws.com/bin/linux/x64/0.5/julia-0.5.1-linux-x86_64.tar.gz | tar -C /opt/julia-0.5.1 -x -z --strip-components=1 -f -
ENV PYTHON /usr/bin/python
ENV TF_USE_GPU 1
ADD setup.jl .
RUN /opt/julia-0.5.1/bin/julia setup.jl
RUN echo "\nPATH=/opt/julia-0.5.1/bin:\$PATH\n" >> /root/.bashrc
CMD ["/opt/julia-0.5.1/bin/julia"]
