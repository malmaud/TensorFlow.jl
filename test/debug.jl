using TensorFlow
using Base.Test

sess = TensorFlow.Session()

a = TensorFlow.constant([NaN])
@test run(sess, TensorFlow.is_nan(a))[1]
a = TensorFlow.constant([Inf])
@test run(sess, TensorFlow.is_inf(a))[1]
a = TensorFlow.constant([1.])
@test run(sess, TensorFlow.is_finite(a))[1]
