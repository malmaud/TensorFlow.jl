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

cross_entropy = reduce_mean(-reduce_sum(y_.*log(y_conv), reduction_indices=[2]))

train_step = train.minimize(train.AdamOptimizer(1e-4), cross_entropy)

correct_prediction = indmax(y_conv, 2) .== indmax(y_, 2)

accuracy = reduce_mean(cast(correct_prediction, Float32))

run(session, initialize_all_variables())

for i in 1:1000
    batch = next_batch(loader, 50)
    if i%100 == 1
        train_accuracy = run(session, accuracy, Dict(x=>batch[1], y_=>batch[2], keep_prob=>1.0))
        info("step $i, training accuracy $train_accuracy")
    end
    run(session, train_step, Dict(x=>batch[1], y_=>batch[2], keep_prob=>.5))
end

testx, testy = load_test_set()
test_accuracy = run(session, accuracy, Dict(x=>testx, y_=>testy, keep_prob=>1.0))
info("test accuracy $test_accuracy")
