# TensorFlow

[![Build Status](https://travis-ci.org/malmaud/TensorFlow.jl.svg?branch=master)](https://travis-ci.org/malmaud/TensorFlow.jl)

A wrapper around [TensorFlow](https://www.tensorflow.org/), a popular open source machine learning framework from Google.

## Basic usage

```julia

using TensorFlow

sess = TensorFlow.Session()

x = TensorFlow.constant(Float64[1,2])
y = TensorFlow.Variable(Float64[3,4])
z = TensorFlow.placeholder(Float64)

w = exp(x + z + -y)

run(sess, TensorFlow.initialize_all_variables())
res = run(sess, w, Dict(z=>Float64[1,2]))
@test res[1] ≈ exp(-1)
```

## Logistic regression example

Realistic demonstration of using variable scopes and advanced optimizers

```julia
using Distributions

# Generate some synthetic data
x = randn(100, 50)
w = randn(50, 10)
y_prob = exp(x*w)
y_prob ./= sum(y_prob,2)

function draw(probs)
    y = zeros(size(probs))
    for i in 1:size(probs, 1)
        idx = rand(Categorical(probs[i, :]))
        y[i, idx] = 1
    end
    return y
end

y = draw(y_prob)

# Build the model
sess = Session(Graph())
X = placeholder(Float64)
Y_obs = placeholder(Float64)

variable_scope("logisitic_model", initializer=Normal(0, .001)) do
    global W = get_variable("weights", [50, 10], Float64)
    global B = get_variable("bias", [10], Float64)
end

Y=nn.softmax(X*W + B)
Loss = -reduce_sum(log(Y).*Y_obs)
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)

# Run training
run(sess, initialize_all_variables())

for epoch in 1:100
    cur_loss, _ = run(sess, vcat(Loss, minimize_op), Dict(X=>x, Y_obs=>y))
    println(@sprintf("Current loss is %.2f.", cur_loss))
end
```

## Limitations

* Only supports Linux and OS X
* Included TensorFlow binaries do not support GPUs (fix coming soon)
