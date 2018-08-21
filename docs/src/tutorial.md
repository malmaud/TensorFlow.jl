# MNIST tutorial

This tutorial is strongly based on the [official TensorFlow MNIST tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html). We will classify MNIST digits, at first using simple logistic regression and then with a deep convolutional model.

Read through the official tutorial! Only the differences from the Python version are documented here.

## Load MNIST data

The `DataLoader` API provided in `"examples/mnist_loader.jl"` has some simple code for loading the MNIST dataset, based on the `MNIST.jl` package.

```julia
include(Pkg.dir("TensorFlow", "examples", "mnist_loader.jl"))
loader = DataLoader()
```

## Start TensorFlow session

```julia
using TensorFlow
sess = Session()
```

## Building a softmax regression model

### Placeholders

```julia
x = placeholder(Float32)
y_ = placeholder(Float32)
```

### Variables

```julia
W = Variable(zeros(Float32, 784, 10))
b = Variable(zeros(Float32, 10))

run(sess, global_variables_initializer())
```

### Predicted Class and Loss Function

```julia
y = nn.softmax(x*W + b)
cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y), axis=[2]))
```

Note several differences from the Python version of the tutorial:

* Python uses `tf.matmul` for matrix multiplication and `*` for element-wise multiplication of tensors in the computation graph. Julia uses `*` for matrix multiplication and `.*` for element-wise multiplication.

* The reduction index for the loss term is 1 in the Python version, but the Julia API assumes 1-based indexing to be consistent with the rest of Julia and so 2 is used.

### Train the model

```julia
train_step = train.minimize(train.GradientDescentOptimizer(.00001), cross_entropy)
for i in 1:1000
    batch = next_batch(loader, 100)
    run(sess, train_step, Dict(x=>batch[1], y_=>batch[2]))
end
```

### Evaluate the model

```julia
correct_prediction = argmax(y, 2) .== argmax(y_, 2)
accuracy=reduce_mean(cast(correct_prediction, Float32))
testx, testy = load_test_set()

println(run(sess, accuracy, Dict(x=>testx, y_=>testy)))
```

## Build a multi-layer convolutional network

There are no significant differences from the Python version, so the entire code is presented here:

```julia
# using TensorFlow
using Distributions
include("mnist_loader.jl")

loader = DataLoader()

session = Session(Graph())

function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end

function bias_variable(shape)
    initial = fill(Float32(.1), shape...)
    return Variable(initial)
end

function conv2d(x, W)
    nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

x = placeholder(Float32)
y_ = placeholder(Float32)

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = reshape(x, [-1, 28, 28, 1])

h_conv1 = nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = reshape(h_pool2, [-1, 7*7*64])
h_fc1 = nn.relu(h_pool2_flat * W_fc1 + b_fc1)

keep_prob = placeholder(Float32)
h_fc1_drop = nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = nn.softmax(h_fc1_drop * W_fc2 + b_fc2)

cross_entropy = reduce_mean(-reduce_sum(y_.*log(y_conv), axis=[2]))

train_step = train.minimize(train.AdamOptimizer(1e-4), cross_entropy)

correct_prediction = argmax(y_conv, 2) .== argmax(y_, 2)

accuracy = reduce_mean(cast(correct_prediction, Float32))

run(session, global_variables_initializer())

for i in 1:1000
    batch = next_batch(loader, 50)
    if i%100 == 1
        train_accuracy = run(session, accuracy, Dict(x=>batch[1], y_=>batch[2], keep_prob=>1.0))
        @info("step $i, training accuracy $train_accuracy")
    end
    run(session, train_step, Dict(x=>batch[1], y_=>batch[2], keep_prob=>.5))
end

testx, testy = load_test_set()
test_accuracy = run(session, accuracy, Dict(x=>testx, y_=>testy, keep_prob=>1.0))
@info("test accuracy $test_accuracy")
```
