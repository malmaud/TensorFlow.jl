using PyCall
using Base.Test

# Try your first TensorFlow program
# https://github.com/tensorflow/tensorflow

@pyimport tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
@test isa(hello, PyObject)

sess = tf.Session()
result = sess[:run](hello)
@test isa(result, String)
@test "Hello, TensorFlow!" == result

a = tf.constant(10)
b = tf.constant(32)
result = sess[:run](a[:__add__](b))
@test isa(result, Array{Int32,0})
@test 42 == result[1]
