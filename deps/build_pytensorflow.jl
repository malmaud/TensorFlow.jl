using Requests

if is_apple()
    url = "https://malmaud.github.io/files/mac/tensorflow.zip"
    run(`sudo easy_install --upgrade six`)
elseif is_unix()
    url = "https://malmaud.github.io/files/linux/tensorflow.zip"
end

base = dirname(@__FILE__)
info("Downloading Python tensorflow wheel")
r = Requests.get(url)
info("Done downloading")
run(`mkdir -p $base/edownloads`)
open("$base/downloads/tensorflow.zip", "w") do file
    write(file, r.data)
end

run(`unzip -o $base/downloads/tensorflow.zip`)
run(`sudo pip install $base/tensorflow-0.10.0rc0-py2-none-any.whl`)

run(`rm -f $base/tensorflow-0.10.0rc0-py2-none-any.whl`)
run(`rm -f $base/libc_api.so`)
run(`rm -f $base/libtensorflow.so`)
