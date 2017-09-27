using PyCall
using Conda

const cur_version = "1.2.0"
const cur_py_version = "1.2.0"



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
bin_dir = joinpath(base, "usr/bin")

if !isdir(download_dir)
    mkdir(download_dir)
end

if !isdir(bin_dir)
    run(`mkdir -p $bin_dir`)
end


function download_and_unpack(url)
    tensorflow_zip_path = joinpath(base, "downloads/tensorflow.zip")
    # Download
    download(url, tensorflow_zip_path)
    # Unpack
    try
        run(`unzip -o $(tensorflow_zip_path)`)
    catch err
        if !isfile(joinpath(base, "libtensorflow_c.so"))
            throw(err)
        else
            warn("Problem unzipping: $err")
        end
    end
end


@static if is_apple()
    download_and_unpack("https://storage.googleapis.com/malmaud-stuff/tensorflow_mac_$cur_version.zip")
    mv("libtensorflow.so", "usr/bin/libtensorflow.dylib", remove_destination=true)
end

@static if is_linux()
    if use_gpu
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_$cur_version.zip"
    else
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_cpu_$cur_version.zip"
    end
    download_and_unpack(url)
    mv("libtensorflow.so", "usr/bin/libtensorflow.so", remove_destination=true)
end



# When TensorFlow issue #8669 is closed and a new version is released, use the official release binaries
# of the TensorFlow C library.
# see https://github.com/tensorflow/tensorflow/issues/8669
# and then replacing the blocks above in this section below with:
#=
function download_and_unpack(url)
    tensorflow_zip_path = joinpath(base, "downloads/tensorflow.zip")
    # Download
    download(url, tensorflow_zip_path)
    # Unpack
    try
        run(`tar -xvzf $(tensorflow_zip_path) --strip-components=2 ./lib/libtensorflow.so`)
    catch err
        if !isfile(joinpath(base, "libtensorflow_c.so"))
            throw(err)
        else
            warn("Problem unzipping: $err")
        end
    end
end

@static if is_apple()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-darwin-x86_64-$cur_version.tar.gz"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("libtensorflow.so", "usr/bin/libtensorflow.dylib", remove_destination=true)
end

@static if is_linux()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$cur_version.tar.gz"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("libtensorflow.so", "usr/bin/libtensorflow.so", remove_destination=true)
end
=#
