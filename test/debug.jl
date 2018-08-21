using TensorFlow
using Test

sess = TensorFlow.Session()

a = TensorFlow.constant([NaN])
@test run(sess, TensorFlow.is_nan(a))[1]
a = TensorFlow.constant([Inf])
@test run(sess, TensorFlow.is_inf(a))[1]
a = TensorFlow.constant([1.])
@test run(sess, TensorFlow.is_finite(a))[1]

let
    x = constant(0)
    y = print(x, [x], message="test", first_n=3, summarize=5)
    # Not sure how to test TensorFlow logging output;
    # just make sure it doesn't run with errors for now
    run(sess, y)
end
