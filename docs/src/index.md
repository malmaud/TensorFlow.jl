# TensorFlow.jl

## Introduction

TensorFlow.jl is a wrapper around [TensorFlow](https://www.tensorflow.org/), a powerful library from Google for implementing state-of-the-art deep-learning models. See [the intro tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html) from Google to get a sense of how TensorFlow works - TensorFlow.jl has a similar API to the Python TensorFlow API described in the tutorials. Then see [the Julia equivalent of that tutorial](tutorial.md).

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

CUDA 8.0 and cudnn are required for GPU usage. For GPU support on Mac OS X, see documentation on [building TensorFlow from source](build_from_source.md).

## Comparison to Python API

The wrapper sticks closely to the Python API and so should be easy to pick up for anyone used to the Python API to pick up. Most function names and arguments are semantically the same.

Some differences:

* When the Python API uses an object-oriented notation like `session.run(node)`, the Julia version would be `run(session, node)`.

* When the Python API asks for a TensorFlow type such as `TensorFloat.float32`, instead pass in a native Julia type like `Float32`.

* Many basic Julia mathematical functions are extended to take a TensorFlow node and return another node representing the delayed execution of that function. For example, `sqrt(constant(4.0))` will return a `Operation` which, when evaluated, returns `2.0`.

## What functionality of TensorFlow is exposed

Currently, a large fraction of the computation graph-building functionality is present. This includes

* All basic unary and binary mathematical functions, such as `sqrt`, `*` (scalar and matrix), etc.

* The most frequently used neural network operations, including convolution, recurrent neural networks with GRU cells, and dropout.

* Neural network trainers, such as `AdamOptimizer`.

* Basic image-loading and resizing operations

* Control flow operations (`while` loops, etc)

* PyBoard graph visualization

Currently not wrapped:

* Distributed graph execution


## Limitations

Since the TensorFlow API is so large, not everything is currently wrapped. If you come across TensorFlow functionality provided by the Python API not available in the Julia API, please file an issue (or even better, submit a pull request). Additionally:

* The Python variable checkpointer and Julia checkpointer use different formats for the checkpoint file, since the Python format is proprietary. The TensorFlow developers have stated that they eventually settle on a format and document it, at which point Julia and Python-trained models can share parameters.
