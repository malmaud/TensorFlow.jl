FROM julia:0.5.1
RUN apt-get update -y && apt-get install -y make gcc g++ bzip2 hdf5-tools unzip libzmq3
WORKDIR /
ADD setup.jl .
RUN julia setup.jl
ADD .bashrc /root
ADD .jupyter /root/.jupyter
EXPOSE 8888
