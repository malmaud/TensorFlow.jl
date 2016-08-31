import Requests

base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")

if !isdir(download_dir)
    mkdir(download_dir)
end

@static if is_apple()
    r = Requests.get("https://storage.googleapis.com/malmaud-stuff/tensorflow-mac.zip")
    open(joinpath(base, "downloads/tensorflow.zip"), "w") do file
        write(file, r.data)
    end
    run(`unzip -o $base/downloads/tensorflow.zip`)
    run(`mkdir -p $base/bazel-out/local-fastbuild/bin/tensorflow/c`)
    mv("libtensorflow.so", "$base/bazel-out/local-fastbuild/bin/tensorflow/libtensorflow.dylib", remove_destination=true)
    mv("libc_api.so", "$base/bazel-out/local-fastbuild/bin/tensorflow/c/libc_api.dylib", remove_destination=true)
end

@static if is_linux()
    if "TF_USE_GPU" ∈ keys(ENV) && ENV["TF_USE_GPU"] == "1"
        info("Building TensorFlow.jl for use on the GPU")
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow-linux.zip"
    else
        info("Building TensorFlow.jl for CPU use only. To enable the GPU, set the TF_USE_GPU environment variable to 1 and rebuild TensorFlow.jl").
        url = "https://storage.googleapis.com/malmaud-stuff/tensorflow_linux_cpu.zip"
    end
    r = Requests.get(url)
    open(joinpath(base, "downloads/tensorflow.zip"), "w") do file
        write(file, r.data)
    end
    run(`unzip -o $base/downloads/tensorflow.zip`)
    run(`mkdir -p $base/bazel-out/local_linux-opt/bin/tensorflow/c`)

    mv("libtensorflow.so", "$base/bazel-out/local_linux-opt/bin/tensorflow/libtensorflow.so", remove_destination=true)
    mv("libc_api.so", "$base/bazel-out/local_linux-opt/bin/tensorflow/c/libc_api.so", remove_destination=true)
end
