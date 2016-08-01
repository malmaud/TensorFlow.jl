import Requests

base = joinpath(Pkg.dir("TensorFlow"), "deps")

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
    mv("libtensorflow.so", "$base/bazel-out/local_linux-fastbuild/bin/tensorflow/libtensorflow.so", remove_destination=true)
    mv("libc_api.so", "$base/bazel-out/local_linux-fastbuild/bin/tensorflow/c/libc_api.so", remove_destination=true)
end
