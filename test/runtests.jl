using TensorFlow
using Base.Test

sess = TensorFlow.Session()

x = TensorFlow.constant(Float64[1,2])
y = TensorFlow.Variable(Float64[3,4])
z = TensorFlow.placeholder(Float64)

w = exp(x + z + -y.var_node)

run(sess, TensorFlow.initialize_all_variables())
res = run(sess, w, Dict(z=>Float64[1,2]))
res_array = convert(Array, res)
@test res_array[1] ≈ exp(-1)
