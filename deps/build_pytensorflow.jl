using Requests

if is_apple()
    conda_url = "https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh"
elseif is_unix()
    conda_url = "https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh"
end

base = dirname(@__FILE__)

run(`mkdir -p $base/downloads`)

info("Downloading miniconda")
run(`wget $conda_url -O $base/downloads/miniconda.sh`)
run(`/bin/bash $base/downloads/miniconda.sh -f -b -p $base/miniconda`)

py_path = joinpath(base, "miniconda", "bin")
run(`$py_path/conda upgrade -y setuptools`)
run(`$py_path/pip install --upgrade --ignore-installed tensorflow`)


ENV["PYTHON"] = "$py_path/python"
Pkg.build("PyCall")
