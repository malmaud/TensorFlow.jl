using PyCall
using Conda
using SHA

const cur_version = "1.10.0" # change checksum while changing version
const cur_py_version = "1.10.0"


############################
# Error message for Windows
############################

if Sys.iswindows()
    error("TensorFlow.jl does not support Windows. Please see https://github.com/malmaud/TensorFlow.jl/issues/204")
end

############################
# Determine if using GPU
############################

use_gpu = "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"

if Sys.isapple() && use_gpu
    @warn("No support for TF_USE_GPU on OS X - to enable the GPU, build TensorFlow from source. Falling back to CPU")
    use_gpu=false
end

if use_gpu
    @info("Building TensorFlow.jl for use on the GPU")
else
    @info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
end



#############################
# Install Python TensorFlow
#############################

if PyCall.conda
    Conda.add_channel("conda-forge")
    Conda.add("tensorflow=" * cur_py_version)
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
    - make sure you run the right pip, for the instance of python that PyCall is looking at.
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

mkpath(download_dir)
mkpath(lib_dir)
mkpath(bin_dir)


function download_and_unpack(url, checksum)
    tensorflow_zip_path = joinpath(download_dir, "tensorflow.tar.gz")

    use_cache = false
    isfile(tensorflow_zip_path) && open(tensorflow_zip_path) do f
        use_cache = bytes2hex(sha256(f)) == checksum
    end
        
    if use_cache
        @info "using cached $tensorflow_zip_path"
    else
        @info "downloading tensorflow from $url"
        download(url, tensorflow_zip_path)
    end
    run(`tar -xzf $tensorflow_zip_path -C downloads`)
end

@static if Sys.isapple()
    if use_gpu
        @warn("No support for TF_USE_GPU on OS X - to enable the GPU, build TensorFlow from source. Falling back to CPU")
        use_gpu = false
    end
    url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-$cur_version.tar.gz"
    checksum = "a9d895ca6a974d8a946e26477339a80fb635419707562200bbcc7352cf71e086" # sha256sum
    download_and_unpack(url, checksum)
    mv("$lib_dir/libtensorflow.so", "$bin_dir/libtensorflow.dylib", force=true)
    mv("$lib_dir/libtensorflow_framework.so", "$bin_dir/libtensorflow_framework.so", force=true)
end

@static if Sys.islinux()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$cur_version.tar.gz"
        checksum = "96074aa88aa74fd0a70a864166e218a2471b00a0504b9f5c963dfadb99d6c66c" # sha256sum
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$cur_version.tar.gz"
        checksum = "abf3baa49d460a2f087bc034d5e65f765817f6d9eede564fd848fef616bb4b87" # sha256sum
    end
    download_and_unpack(url, checksum)
    mv("$lib_dir/libtensorflow.so", "$bin_dir/libtensorflow.so", force=true)
    mv("$lib_dir/libtensorflow_framework.so", "$bin_dir/libtensorflow_framework.so", force=true)
end
