# Basic usage

```julia
using TensorFlow

sess = Session()

x = constant(Float64[1,2])
y = Variable(Float64[3,4])
z = placeholder(Float64)

w = exp(x + z + -y)

run(sess, global_variables_initializer())
res = run(sess, w, Dict(z=>Float64[1,2]))
Base.Test.@test res[1] ≈ exp(-1)
```
