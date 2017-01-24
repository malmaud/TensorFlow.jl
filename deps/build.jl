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

function try_unzip()
    try
        run(`unzip -o $base/downloads/tensorflow.zip`)
    catch err
        if !isfile(joinpath(base, "libtensorflow_c.so"))
            throw(err)
        else
            warn("Problem unzipping: $err")
        end
    end
end

@static if is_apple()
    r = Requests.get("https://storage.googleapis.com/malmaud-stuff/tensorflow_mac_v4.zip")
    open(joinpath(base, "downloads/tensorflow.zip"), "w") do file
        write(file, r.data)
    end
    try_unzip()
    mv("libtensorflow_c.so", "usr/bin/libtensorflow_c.dylib", remove_destination=true)
end

@static if is_linux()
    if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
        info("Building TensorFlow.jl for use on the GPU")
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_v5.zip"
    else
        info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl")
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_cpu_v4.zip"
    end
    r = Requests.get(url)
    open(joinpath(base, "downloads/tensorflow.zip"), "w") do file
        write(file, r.data)
    end
    try_unzip()
    mv("libtensorflow_c.so", "usr/bin/libtensorflow_c.so", remove_destination=true)
end
