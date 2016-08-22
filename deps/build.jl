import Requests

base = dirname(@__FILE__)
download_dir = joinpath(base, "downloads")

if !isdir(download_dir)
    mkdir(download_dir)
end

@static if is_apple()
    r = Requests.get("https://malmaud.github.io/files/mac/tensorflow.zip")
    open(joinpath(base, "downloads/tensorflow.zip"), "w") do file
        write(file, r.data)
    end
    run(`unzip -o $base/downloads/tensorflow.zip`)
    run(`mkdir -p $base/bazel-out/local-fastbuild/bin/tensorflow/c`)
    mv("libtensorflow.so", "$base/bazel-out/local-fastbuild/bin/tensorflow/libtensorflow.dylib", remove_destination=true)
    mv("libc_api.so", "$base/bazel-out/local-fastbuild/bin/tensorflow/c/libc_api.dylib", remove_destination=true)
end

@static if is_linux()
    r = Requests.get("https://malmaud.github.io/files/linux/tensorflow.zip")
    open(joinpath(base, "downloads/tensorflow.zip"), "w") do file
        write(file, r.data)
    end
    run(`unzip -o $base/downloads/tensorflow.zip`)
    run(`mkdir -p $base/bazel-out/local_linux-fastbuild/bin/tensorflow/c`)
    run(`mkdir -p $base/bazel-out/local_linux-opt/bin/tensorflow/c`)
    mv("libtensorflow.so", "$base/bazel-out/local_linux-fastbuild/bin/tensorflow/libtensorflow.so", remove_destination=true)
    mv("libc_api.so", "$base/bazel-out/local_linux-fastbuild/bin/tensorflow/c/libc_api.so", remove_destination=true)
    mv("libtensorflow_gpu.so", "$base/bazel-out/local_linux-opt/bin/tensorflow/libtensorflow.so", remove_destination=true)
    mv("libc_api_gpu.so", "$base/bazel-out/local_linux-opt/bin/tensorflow/c/libc_api.so", remove_destination=true)
end
