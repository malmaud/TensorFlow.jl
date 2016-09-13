# Shape inference

Shape inference allows you to statically query the shape of a tensor before the computation graph has been run. This lets you verify that tensors have their expected shape. Invoke it by calling `get_shape` on a `Tensor` object.

The Python API to TensorFlow implements shape inference in the Python layer. As it is not part of the C layer, it has been partially reimplemented in Julia. The Julia implementation supports shape inference most of the common operations. Contributing to shape inference is easy and pull requests are welcome - just look at the self-contained "shape_inference.jl" file for examples.
