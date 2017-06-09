
base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")
bin_dir = joinpath(base, "usr/bin")

if !isdir(download_dir)
    mkdir(download_dir)
end

if !isdir(bin_dir)
    run(`mkdir -p $bin_dir`)
end

# When TensorFlow 1.1 is released, use the official release binaries
# of the TensorFlow C library. Do this by setting cur_version to 1.1.0
# and then replacing the blocks below with:
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
    if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
        info("Building TensorFlow.jl for use on the GPU")
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-darwin-x86_64-$cur_version.tar.gz"
    else
        info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("libtensorflow.so", "usr/bin/libtensorflow.dylib", remove_destination=true)
end

@static if is_linux()
    if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
        info("Building TensorFlow.jl for use on the GPU")
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$cur_version.tar.gz"
    else
        info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
        url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$cur_version.tar.gz"
    end
    download_and_unpack(url)
    mv("libtensorflow.so", "usr/bin/libtensorflow.so", remove_destination=true)
end
=#

const cur_version = "1.1.0"


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
    if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
        info("Building TensorFlow.jl for use on the GPU")
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_$cur_version.zip"
    else
        info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_cpu_$cur_version.zip"
    end
    download_and_unpack(url)
    mv("libtensorflow.so", "usr/bin/libtensorflow.so", remove_destination=true)
end
