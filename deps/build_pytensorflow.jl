using Requests

# For Travis
if is_apple()
    py_path = "/usr/bin"
elseif is_unix()
    py_path = "/opt/python/2.7.10/bin"
end

if is_apple()
    url = "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py2-none-any.whl"
    wheel_name = "tensorflow-0.10.0rc0-py2-none-any.whl"
    run(`sudo $py_path/easy_install --upgrade six`)
elseif is_unix()
    wheel_name = "tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl"
    url = "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl"
end

base = dirname(@__FILE__)
info("Downloading Python tensorflow wheel")
r = Requests.get(url)
info("Done downloading")
open("$base/wheel_name", "w") do file
    write(file, r.data)
end
run(`sudo $py_path/pip install $base/$wheel_name`)

run(`rm -f $base/$wheel_name`)
run(`rm -f $base/libc_api.so`)
run(`rm -f $base/libtensorflow.so`)
