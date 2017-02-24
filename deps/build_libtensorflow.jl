#=
Script for building the Linux version of libtensorflow_c.so.

Requires Docker.
=#

if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
    docker_image = "tensorflow/tensorflow:1.0.0-devel-gpu"
    opts = "-c opt --config=cuda"
    info("Building libtensorflow_c.so for use on the GPU.")
else
    docker_image = "tensorflow/tensorflow:1.0.0-devel"
    opts = "-c opt"
    info("Building libtensorflow_c.so for use on the CPU. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rerun this script.")
end

build_script = """
cd /tensorflow;
bazel build $opts //tensorflow:libtensorflow_c.so
"""

run(`docker pull $docker_image`)

##################
# Ensure we pick a name for the container not already in use.
container_pids = readstring(`docker ps -aq --no-trunc`)
names = []

for pid in split(container_pids, "\n")
    isempty(pid) && continue
    name = readstring(`docker inspect --format={{.Name}} $pid`)
    push!(names, chomp(name))
end

for i in countfrom()
    global container_name = "/tf_$i"
    container_name ∈ names || break
end
##################

run(`docker run --name=$container_name $docker_image /bin/bash -c $build_script`)

run(`docker cp $(container_name[2:end]):/tensorflow/bazel-bin/tensorflow/libtensorflow_c.so usr/bin`)
