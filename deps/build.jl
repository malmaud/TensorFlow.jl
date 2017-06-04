import Requests

base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")
bin_dir = joinpath(base, "usr/bin")

if !isdir(download_dir)
    mkdir(download_dir)
end

if !isdir(bin_dir)
    run(`mkdir -p $bin_dir`)
end

const cur_version = "1.1.0"

function try_unzip()
    try
        run(`tar -xvzf $base/downloads/tensorflow.tar.gz --strip-components=2 ./lib/libtensorflow.so`)
    catch err
        if !isfile(joinpath(base, "libtensorflow.so"))
            throw(err)
        else
            warn("Problem extracting $err")
        end
    end
end

if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
    info("Building TensorFlow.jl for use on the GPU")
    processor = "gpu"
else
    info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
    processor = "cpu"
end

@static if is_apple()
    ext = "dylib"
    os = "darwin"
end

@static if is_linux()
    ext = "so"
    os = "linux"
end

url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-$processor-$os-x86_64-$cur_version.tar.gz"

r = Requests.get(url)
open(joinpath(base, "downloads/tensorflow.tar.gz"), "w") do file
    write(file, r.data)
end
try_unzip()
mv("libtensorflow.so", "usr/bin/libtensorflow.$ext", remove_destination=true)
