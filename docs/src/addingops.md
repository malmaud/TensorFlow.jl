TensorFlow has a huge number of operations.
TensorFlow.jl implements many of them, but there are many more still to be added.
Linking up to the TensorFlow API to implement a new operation is a fun and easy way to contribute.

See also the [TensorFlow guide for writing language bindings.](https://www.tensorflow.org/extend/language_bindings).


Once you have identified the operation you want to add, the following steps should be taken.
(If you like [Test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development), feel free to bring the tests first.)

 1. Implement the operation:
     - This normally means adding it to one of the files `/src/ops/`.
     - You can find the operation's "true name", and its input and attribute definitions by looking in the [TensorFlow ops list](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt).
     - You can base the implementation on something already there.
     - Don't forget to shift the indexes to be 1 based, as TensorFlow defaults to zero based.
     - Export your new op by adding it to the exports list in `/src/TensorFlow.jl`.
 2. Add the doc string:
     - You can normally base this directly off the docs for Python.
     - You can find the python docs either by finding the `OpName.md` file, or using the summary and description from the afformentioned op list.
     - Don't forget to change the examples to Julia, and to reindex them.
 3. Implement some tests:
     - You can often base these off the examples from the docs for the operation.
     - the test filename normally matches to the name of the soure file from the `/src/ops` directory, but in `/test`.
     - For bonus points include a test which requires the gradient to be successfully calculated through your new op -- sometime the op works but for one reason or another its gradient does not.
 4. Implement shape inference:
     - The op needs to be added to `src/shape_inference.jl`.
     - This is done by `register_shape("true_name")`, where `true_name` is the name from the [TensorFlow ops list](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt).
     - Again, you can often base it off a method already written.
     - The docs for the op are helpful in determining the rules for the output shape -- it is often given explictly (if sometimes cryptically).
     - If really confused as to how the shape is determined, you can use your op and check the shapes by running the operators .
        - eg `size(run(sess, myop(constant(...)))`.
        - You can a generate large enough quantity of these results: you can write a pile of tests so that if they pass, you are certain the inference is right.
        - Then you can tweak and permute your shape inference code, until it passes.
 5. Implement shape inference tests:
     - These go in `test/shape_inference.jl`.
     
     
     
