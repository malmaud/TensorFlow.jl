using PyCall
using Conda

const cur_version = "1.2.0"
const cur_py_version = "1.2.0"

############################
# Determine if using GPU
############################

use_gpu = "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"

if is_apple() && use_gpu
    warn("No support for TF_USE_GPU on OS X - to enable the GPU, build TensorFlow from source. Falling back to CPU")
    use_gpu=false
end

if is_windows() && pyversion<v"3.5.0"
    error("On windows Python 3.5 or better is required. PyCall is currently using $(pyversion), please rebuild PyCall to use a newer version of Python.")
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
mkpath(download_dir)
mkpath(bin_dir)

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


@static if is_windows()
    url = "http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-windows-x86_64.zip"
    if use_gpu
        # This is not the correct location.
        # So error will probably happen here.
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-$cur_version.zip"
    else
      url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-$cur_version.zip"
    end

    tensorflow_zip_path = joinpath(base, "downloads/tensorflow.zip")
    # Download
    download(url, tensorflow_zip_path)
    # Unpack

    # Hacky way to do an unzip in surficently up to date versions of windows.
    # From https://stackoverflow.com/a/26843122/179081
    # better is probably to just use ZipFile.jl package
    run(`powershell.exe -nologo -noprofile -command "& { Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('$(tensorflow_zip_path)', '.'); }"`)

    mv("tensorflow.dll", "usr/bin/libtensorflow.dll", remove_destination=true)
end
