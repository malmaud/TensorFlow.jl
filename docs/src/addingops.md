TensorFlow has a huge number of operations.
TensorFlow.jl implements many of them; but their are many more still to be added.
Linking up to the TensorFlow API to implement a new operation is a fun and easy way to contribute.

See also the [TensorFlow guide for writing language bindings.](https://www.tensorflow.org/extend/language_bindings).


Once you have identified the operation you want to add, the following steps should be taken.
(If you like TDD, feel free to bring the tests first)

 1. Implement the opteration
     - This normally means adding it to one of the files `/src/ops/`
     - You can find the operation's "true name", and its input and attrinbute definitions by looking in the [TensorFlow ops list](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt)
     - You can base the implementation on something already there
     - Don't forget to shift indexs to be 1 based, as tensorflow defaults to zero based.\
     - Export your new op by adding it to the export's list in `/src/TensorFlow.jl`
 2. Add the doc string.     
     - You can normally base this directly off the doc for python
     - You can find the python docs either by finding the `OpName.md` file, or using the summary+description from the afformentioned op list
     - Don't forget to change the examples to julia, and to reindex them.
 3. Implement some tests
     - You can often base these of the examples from the docs
     - the test file name normally matchs to the name of the file from the `/src/ops` directory, but in `/test`
     - For bonus points include a test which requires the gradient to be successfully calculated through your new op -- sometime the op works but for one reason or another it's graident does not
 4. Implement shape inference
     - The op needs to be added to `src/shape_inference.jl`
     - This is done by `register_shape("true_name")`, where `true_name` is the name from the [TensorFlow ops list](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt)
     - Again, you can often base it off something already their
     - The docs for the op are helpful in determining the rules for the output shape -- it is often given explictly (if sometimes cryptically)
     - If really confused you can use your op and check the shapes by running the operators 
        - eg `size(run(sess, myop(constant(...)))`. 
        - You can a large enough quantity of these results to write a pile of tests so that if they pass, you are certain the inference is right.
        - Then you can tweak and permute your shape inference code, until it passes.
 5. Implement shape inference tests
     - These go in `test/shape_inference.jl`
     
     
     
