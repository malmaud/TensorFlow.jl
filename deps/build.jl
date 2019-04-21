using PyCall
using Conda

const cur_version = "1.13.1"
const cur_py_version = "1.13.1"


############################
# message for Windows
############################

if Sys.iswindows() && pyversion<v"3.5.0"
    error("On windows Python 3.5 or better is required. PyCall is currently using $(pyversion), please rebuild PyCall to use a newer version of Python.")
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


function download_and_unpack(url)
    tensorflow_zip_path = joinpath(base, "downloads/tensorflow.tar.gz")
    download(url, tensorflow_zip_path)
    run(`tar -xzf $tensorflow_zip_path -C downloads`)
end

@static if Sys.isapple()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-darwin-x86_64-$cur_version.tar.gz"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("$lib_dir/libtensorflow.so", "usr/bin/libtensorflow.dylib", force=true)
    mv("$lib_dir/libtensorflow_framework.so", "usr/bin/libtensorflow_framework.so", force=true)
end

@static if Sys.islinux()
    if use_gpu
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$cur_version.tar.gz"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("$lib_dir/libtensorflow.so", "usr/bin/libtensorflow.so", force=true)
    mv("$lib_dir/libtensorflow_framework.so", "usr/bin/libtensorflow_framework.so", force=true)
end

@static if Sys.iswindows()
    url = "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-windows-x86_64.zip"
    if use_gpu
        # This is not the correct location.
        # So error will probably happen here.
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-$cur_version.zip"
    else
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-$cur_version.zip"
    end

    tensorflow_zip_path = joinpath(download_dir, "tensorflow.zip")
    # Download
    download(url, tensorflow_zip_path)
    # Unpack

    # Hacky way to do an unzip in surficently up to date versions of windows.
    # From https://stackoverflow.com/a/26843122/179081
    # better is probably to just use ZipFile.jl package
    # run(`powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('$(tensorflow_zip_path)', '.'); }"`)
    using InfoZIP
    println(tensorflow_zip_path)
    InfoZIP.unzip(tensorflow_zip_path, download_dir)

    tensorflow_path = joinpath(lib_dir, "tensorflow.dll")

    println(tensorflow_path)
    tf_path = joinpath(joinpath(joinpath(base, "usr"), "bin"), "libtensorflow.dll")
    println(tf_path)
    mv(tensorflow_path, tf_path, force=true)
end