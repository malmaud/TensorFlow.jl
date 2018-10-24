# Building TensorFlow from source

Building TensorFlow from source is recommended by Google for maximum performance, especially when running in CPU mode - on some systems, the difference can be substantial. It will also be required for Mac OS X GPU support for TensorFlow versions later than 1.1. This document describes how to do so.


## Step 1: Build libtensorflow

To build libtensorflow for TensorFlow.jl, follow the [official instructions for building tensorFLow from source](https://www.tensorflow.org/install/install_sources), except for a few minor modifications so as to build the library rather than the client.

  * In the step "Prepare environment", ignore "Install python dependencies" -- these are not necessary as we are not building for Python. Be sure to follow all other steps as needed for your OS.
  * In the step "Build the pip package", since we are building the binary file and not the pip package, instead run `bazel build --config=opt //tensorflow:libtensorflow.so`, adding `--config=cuda` if GPU support is desired.

Running `bazel build` will produce the `libtensorflow.so` binary needed by TensorFlow.jl - there is no need to build the Python package or run anything else. You may place the binary wherever is convenient.
If on Mac OS X, you may need to rename the `libtensorflow.so` to `libtensorflow.dylib`.

## Step 2: Install the TensorFlow binary

We must now tell TensorFlow.jl to where to load the custom binary from. There are a number of ways to do so.

  * We can set the environment variable `LIBTENSORFLOW` to `/path/to/tensorflow.so`. This may be done system-wide by editing `.profile` or any other method supported by your OS.

  * For users of the Atom/Juno IDE who do not wish to modify their system-wide environment, environment variables may be set by adding the line `process.env.LIBTENSORFLOW = "/path/to/libtensorflow.so"` to the `init.coffee` script (easily accessible by clicking `File -> Init Script`). Note that Atom may not always inherit environment variables set by the OS.

  * Or you can copy `libtensorflow.so` to `<TensorFlowDIR>/deps/usr/bin/`, overwriting the included binary. Where `<TensorFlowDIR>` is the path to the directory TensorFlow.jl is being loaded from.
  

## Step 3: Check that the custom binary is loaded

After running `using TensorFlow`, it should no longer complain that TensorFlow wasn't compiled with the necessary instructions. Try generating two random matrices and multiplying them together. You can time the computation with `@time run(sess, x)`, which should be much faster.


## Tips & known issues

  * For maximum performance, you should always compile on the same system that will be running the computation, and with the correct CUDA Compute Capability version supported by your GPU.

  * If TensorFlow.jl fails to load with the error `Library not loaded: @rpath/libcublas.8.0.dylib` or any similar error, it means that the CUDA libraries are not in `LD_LIBRARY_PATH` as required by Nvidia. Be sure to add `/usr/local/cuda/lib`, or wherever your CUDA instalation is located, to `LD_LIBRARY_PATH`. This may be done by editing `.profile`, or for Atom/Juno users editing `init.coffee`, or any other method supported by your OS, as described in Step 2. Be careful that you append this folder and do not mistakenly overwrite your entire path.

  * If you get `CUDA_ERROR_NOT_INITIALIZED`, then for some reason TensorFlow cannot find your GPU. Make sure that the appropriate software is installed, and if using an external GPU, make sure it is plugged in correctly.

  * To check whether the GPU is being used, create your session with `TensorFlow.Session(config=TensorFlow.tensorflow.ConfigProto(log_device_placement=true))`. TensorFlow will then print information about which device is used.

  * You may need to add symlinks from `libcudnn5.dylib` to `libcudnn.5.dylib` so that Bazel is able to correctly locate the necessary dependencies.

  * On Mac OS X, `nvcc`, Nvidia's CUDA compiler, requires OS X Command Line Tools version 8.2 and does not work with the latest version. You can download this version from Apple's website, and switch to it by running `sudo xcode-select -s /path/to/CommandLineTools`.

  * On Mac OS X, make sure to set the environment variable `GCC_HOST_COMPILER_PATH` to `/usr/bin/gcc` - do not install GCC yourself, or the build may fail with obscure error messages.

  * On Mac OS X, if you don't wish to install Homebrew, you can instead use Julia's internal Homebrew-based dependency manager Homebrew.jl by running ```Homebrew.brew(`install --build-from-source libtensorflow`)```. GPU support can be enabled by modifying the Ruby formula using ```Homebrew.brew(`edit libtensorflow`)``` -- you should set all necessary environment variables in the Ruby formula, as Homebrew may not display prompts correctly.
