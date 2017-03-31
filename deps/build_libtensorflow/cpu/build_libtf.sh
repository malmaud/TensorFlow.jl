cd /tensorflow
bazel build -c opt //tensorflow:libtensorflow.so
mkdir -p /out/cpu
cp bazel-bin/tensorflow/libtensorflow.so /out/cpu
