using Requests

if is_apple()
    url = "https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0rc0-py2-none-any.whl"
    wheel_name = "tensorflow-0.12.0rc0-py2-none-any.whl"
    conda_url = "https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh"
elseif is_unix()
    wheel_name = "tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl"
    url = "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl"
    conda_url = "https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh"
end

base = dirname(@__FILE__)

run(`mkdir -p $base/downloads`)

info("Downloading miniconda")
run(`wget $conda_url -O $base/downloads/miniconda.sh`)
run(`/bin/bash $base/downloads/miniconda.sh -f -b -p $base/miniconda`)


info("Downloading Python tensorflow wheel")
r = Requests.get(url)
info("Done downloading")
open("$base/$wheel_name", "w") do file
    write(file, r.data)
end

py_path = joinpath(base, "miniconda", "bin")
run(`$py_path/conda upgrade -y setuptools`)
run(`$py_path/pip install --upgrade --ignore-installed $base/$wheel_name`)
run(`rm -f $base/$wheel_name`)

ENV["PYTHON"] = "$py_path/python"
Pkg.build("PyCall")
