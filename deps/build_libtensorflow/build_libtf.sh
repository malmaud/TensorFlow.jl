echo "Saving files to $prefix"
cd /gpu
docker build -t tensorflowjl:gpu .
cd /cpu
docker build -t tensorflowjl:cpu .
cd /
docker run -v $prefix:/out tensorflowjl:cpu
docker run -v $prefix:/out tensorflowjl:gpu
