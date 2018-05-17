using PyCall
using Conda

const cur_version = "1.7.0"
const cur_py_version = "1.7.0"



############################
# Error message for Windows
############################

if is_windows()
    error("TensorFlow.jl does not support Windows. Please see https://github.com/malmaud/TensorFlow.jl/issues/204")
end

############################
# Determine if using GPU
############################

use_gpu = "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"

if is_apple() && use_gpu
    warn("No support for TF_USE_GPU on OS X - to enable the GPU, build TensorFlow from source. Falling back to CPU")
    use_gpu=false
end

if use_gpu
    info("Building TensorFlow.jl for use on the GPU")
else
    info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
end



#############################
# Install Python TensorFlow
#############################

if PyCall.conda
    Conda.add_channel("conda-forge")
    Conda.add("tensorflow=" * cur_py_version)
    Conda.add("tensorboard=" * cur_py_version)
else
    try
        pyimport("tensorflow")
        # See if it works already
    catch ee
        typeof(ee) <: PyCall.PyError || rethrow(ee)
        error("""
Python TensorFlow not installed
Please either:
 - Rebuild PyCall to use Conda, by running in the julia REPL:
    - `ENV["PYTHON"]=""; Pkg.build("PyCall"); Pkg.build("TensorFlow")`
 - Or install the python binding yourself, eg by running pip
    - `pip install tensorflow`
    - then rebuilding TensorFlow.jl via `Pkg.build("TensorFlow")` in the julia REPL
""")
    end
end


############################
# Install libtensorflow
############################

base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")
lib_dir = joinpath(download_dir, "lib")
bin_dir = joinpath(base, "usr/bin")

if !isdir(download_dir)
    mkdir(download_dir)
end

if !isdir(bin_dir)
    run(`mkdir -p $bin_dir`)
end

function download_and_unpack(url)
    tensorflow_zip_path = joinpath(base, "downloads/tensorflow.tar.gz")
    # Download
    download(url, tensorflow_zip_path)
    # Unpack
    run(`tar -xzf $tensorflow_zip_path -C downloads`)
end

@static if is_apple()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-darwin-x86_64-$cur_version.tar.gz"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("$lib_dir/libtensorflow.so", "usr/bin/libtensorflow.dylib", remove_destination=true)
    mv("$lib_dir/libtensorflow_framework.so", "usr/bin/libtensorflow_framework.so", remove_destination=true)
end

@static if is_linux()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$cur_version.tar.gz"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("$lib_dir/libtensorflow.so", "usr/bin/libtensorflow.so", remove_destination=true)
    mv("$lib_dir/libtensorflow_framework.so", "usr/bin/libtensorflow_framework.so", remove_destination=true)
end

