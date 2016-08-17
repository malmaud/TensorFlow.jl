using Requests

# For Travis
if is_apple()
    py_path = "/usr/bin"
elseif is_unix()
    py_path = "/opt/python/2.7.10/bin"
end

if is_apple()
    url = "https://malmaud.github.io/files/mac/tensorflow.zip"
    run(`sudo $py_path/easy_install --upgrade six`)
elseif is_unix()
    url = "https://malmaud.github.io/files/linux/tensorflow.zip"
end

base = dirname(@__FILE__)
info("Downloading Python tensorflow wheel")
r = Requests.get(url)
info("Done downloading")
run(`mkdir -p $base/downloads`)
open("$base/downloads/tensorflow.zip", "w") do file
    write(file, r.data)
end
cd(base) do
    run(`unzip -o $base/downloads/tensorflow.zip`)
end
run(`sudo $py_path/pip install $base/tensorflow-0.10.0rc0-py2-none-any.whl`)

run(`rm -f $base/tensorflow-0.10.0rc0-py2-none-any.whl`)
run(`rm -f $base/libc_api.so`)
run(`rm -f $base/libtensorflow.so`)
