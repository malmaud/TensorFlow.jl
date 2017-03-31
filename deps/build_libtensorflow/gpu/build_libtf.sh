cd /tensorflow
bazel build -c opt --config=cuda //tensorflow:libtensorflow.so
mkdir -p /out/gpu
cp bazel-bin/tensorflow/libtensorflow.so /out/gpu
