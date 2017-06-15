# Building TensorFlow from source

Building TensorFlow from source is recommended by Google for maximum performance - on some systems, the difference can be substantial. It will also be required for Mac OS X GPU support for versions later than 1.1. This document describes how to do so.


## Step 1: build libtensorflow

To build libtensorflow for TensorFlow.jl, follow the steps here: https://www.tensorflow.org/install/install_sources, except for a few minor modifications.

  * Ignore the step "Install python dependencies".
  * In the step "Build the pip package", since we are building the binary file and not the pip package, instead run `bazel build --config=opt //tensorflow:libtensorflow.so`, adding `--config=cuda` if GPU support is desired.

Running `bazel build` will produce the `libtensorflow.so` binary needed by TensorFlow.jl - there is no need to build the Python package or run anything else. You may place the binary wherever is convenient.


## Step 2: add necessary environment variables

To use a custom TensorFlow binary, we must set the appropriate environment variables so that TensorFlow.jl knows its location, and if GPU support is desired, so that CUDA is loaded correctly.
For users using the Atom/Juno IDE, this may be done by adding the following two lines to the `init.coffee` script (easily accessible by clicking `File -> Init Script`).

  * `process.env.LD_LIBRARY_PATH = "/usr/local/cuda/lib:..."`
  * `process.env.LIBTENSORFLOW = "/path/to/libtensorflow.so"`


## Step 3: check that the custom binary is loaded

After running `using TensorFlow`, it should no longer complain that TensorFlow wasn't compiled with the necessary instructions. Try generating two random matrices and multiplying them together. You can time the computation with `@time run(sess, x)`, which should be much faster.


## Tips & known issues

  * If you encounter segmentation faults or other errors, try `Pkg.checkout("TensorFlow")`.

  * For maximum performance, you should always compile on the same system that will be running the computation, and with the correct CUDA Compute Capability version supported by your GPU.

  * If you get `CUDA_ERROR_NOT_INITIALIZED`, then for some reason TensorFlow cannot find your GPU. Make sure that the appropriate software is installed, and if using an external GPU, make sure it is plugged in correctly.

  * To check whether GPU is being used, create your session with `TensorFlow.Session(config=TensorFlow.tensorflow.ConfigProto(log_device_placement=true))`. TensorFlow will then print information about which device is used.

  * Currently, TensorFlow.jl uses Python interop for `tf.gradients`, because it is not implemented in the TensorFlow C API.  If you encounter issues, you may need to update the `tensorflow` package in Conda.jl by running `Conda.update("tensorflow")`.

  * You may need to add symlinks from `libcudnn5.dylib` to `libcudnn.5.dylib` so that Bazel is able to correctly locate the necessary dependencies.

  * On Mac OS X, `nvcc`, Nvidia's CUDA compiler, requires OS X Command Line Tools version 7.3 and does not work with the latest version. You can download this version from Apple's website, and switch to it by running `sudo xcode-select -s /path/to/CommandLineTools`.

  * On Mac OS X, make sure to set the environment variable `GCC_HOST_COMPILER_PATH` to `/usr/bin/gcc` - do not install GCC yourself, or the build may fail with obscure error messages.

  * On Mac OS X, if you don't wish to install Homebrew, you can instead use Julia's internal Homebrew-based dependency manager Homebrew.jl by running ```Homebrew.brew(`install --build-from-source libtensorflow`)```. GPU support can be enabled by modifying the Ruby formula using ```Homebrew.brew(`edit libtensorflow`)``` -- you should set all necessary environment variables in the Ruby formula, as Homebrew may not display prompts correctly.
