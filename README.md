# TensorFlow

[![Build Status](https://travis-ci.org/malmaud/TensorFlow.jl.svg?branch=master)](https://travis-ci.org/malmaud/TensorFlow.jl)
[![codecov.io](http://codecov.io/github/malmaud/TensorFlow.jl/coverage.svg?branch=master)](http://codecov.io/github/malmaud/TensorFlow.jl?branch=master)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.01002/status.svg)](https://doi.org/10.21105/joss.01002)

A wrapper around [TensorFlow](https://www.tensorflow.org/), a popular open source machine learning framework from Google.

## Documentation

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://malmaud.github.io/TensorFlow.jl/latest)

Other resources:

* [TensorFlow Dev Summit 2019 presentation](https://www.youtube.com/watch?v=n2MwJ1guGVQ&t=2s)
* [JuliaCon 2017 presentation](https://www.youtube.com/watch?v=MaCf1PtHEJo)

## Why use TensorFlow.jl?

See a [list of advantages](https://github.com/malmaud/TensorFlow.jl/blob/master/docs/src/why_julia.md)
over the Python API.

## What's changed recently?

See [NEWS](https://github.com/malmaud/TensorFlow.jl/blob/master/NEWS.md).


## Basic usage

```julia

using TensorFlow
using Test

sess = TensorFlow.Session()

x = TensorFlow.constant(Float64[1,2])
y = TensorFlow.Variable(Float64[3,4])
z = TensorFlow.placeholder(Float64)

w = exp(x + z + -y)

run(sess, TensorFlow.global_variables_initializer())
res = run(sess, w, Dict(z=>Float64[1,2]))
@test res[1] ≈ exp(-1)
```

## Installation

Install via

```julia
Pkg.add("TensorFlow")
```

To enable support for GPU usage on Linux, set an environment variable `TF_USE_GPU` to "1" and then rebuild the package. eg

```julia
ENV["TF_USE_GPU"] = "1"
Pkg.build("TensorFlow")
```

CUDA 8.0 and cudnn are required for GPU usage.
If you need to use a different version of CUDA, or if you want GPU support on Mac OS X, you can [compile libtensorflow from source](#optional-using-a-custom-tensorflow-binary).

Initial precompilation (eg, the first time you type `using TensorFlow`) can take around five minutes, so please be patient. Subsequent load times will only be a few seconds.

## Installation via Docker

Simply run `docker run -it malmaud/julia:tf` to open a Julia REPL that already
has TensorFlow installed:

```julia
julia> using TensorFlow
julia>
```

For a version of TensorFlow.jl that utilizes GPUs, use `nvidia-docker run -it malmaud/julia:tf_gpu`.
Download [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you don't
already have it.

## Logistic regression example

Realistic demonstration of using variable scopes and advanced optimizers

```julia
using TensorFlow
using Distributions
using Printf

# Generate some synthetic data
x = randn(100, 50)
w = randn(50, 10)
y_prob = exp.(x*w)
y_prob ./= sum(y_prob,dims=2)

function draw(probs)
    y = zeros(size(probs))
    for i in 1:size(probs, 1)
        idx = rand(Categorical(probs[i, :]))
        y[i, idx] = 1
    end
    return y
end

y = draw(y_prob)

# Build the model
sess = Session(Graph())

X = placeholder(Float64, shape=[-1, 50])
Y_obs = placeholder(Float64, shape=[-1, 10])

variable_scope("logisitic_model"; initializer=Normal(0, .001)) do
    global W = get_variable("W", [50, 10], Float64)
    global B = get_variable("B", [10], Float64)
end

Y=nn.softmax(X*W + B)

Loss = -reduce_sum(log(Y).*Y_obs)
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)
saver = train.Saver()

# Run training
run(sess, global_variables_initializer())
checkpoint_path = mktempdir()
@info("Checkpoint files saved in $checkpoint_path")
for epoch in 1:100
    cur_loss, _ = run(sess, [Loss, minimize_op], Dict(X=>x, Y_obs=>y))
    println(@sprintf("Current loss is %.2f.", cur_loss))
    train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)
end
```



## Troubleshooting

If you see issues from the ccall or python interop, try updating TensorFlow both in Julia and in the global python install:

```julia
julia> Pkg.build("TensorFlow")
```

```bash
$ pip install --upgrade tensorflow
```

## Citing

If you use this software in your research, we would really appreciate if you cite us.

```
Malmaud, J. & White, L. (2018). TensorFlow.jl: An Idiomatic Julia Front End for TensorFlow. Journal of Open Source Software, 3(31), 1002, https://doi.org/10.21105/joss.01002
```

## Optional: Using a custom TensorFlow binary

To build TensorFlow from source, or if you already have a TensorFlow binary that you wish to use, follow [these instructions](https://malmaud.github.io/TensorFlow.jl/latest/build_from_source.html). This is recommended by Google for maximum performance, and is currently needed for Mac OS X GPU support.

For Linux users, a convenience script is included to use Docker to easily build the library. Just install docker and run `julia build_libtensorflow.so` from the "deps" directory of the TensorFlow.jl package. Note that this method may not link to all libraries available on the target system such as Intel MKL.
