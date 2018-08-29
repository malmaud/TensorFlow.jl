# using TensorFlow
include("mnist_loader.jl")

loader = DataLoader()

sess = Session(Graph())

x = placeholder(Float32)
y_ = placeholder(Float32)

W = Variable(zeros(Float32, 784, 10))
b = Variable(zeros(Float32, 10))

run(sess, global_variables_initializer())

y = nn.softmax(x*W + b)

cross_entropy = reduce_mean(-reduce_sum(y_ .* log(y), axis=[2]))
train_step = train.minimize(train.GradientDescentOptimizer(.00001), cross_entropy)

correct_prediction = argmax(y, 2) .== argmax(y_, 2)
accuracy=reduce_mean(cast(correct_prediction, Float32))

for i in 1:1000
    batch = next_batch(loader, 100)
    run(sess, train_step, Dict(x=>batch[1], y_=>batch[2]))
end

testx, testy = load_test_set()

println(run(sess, accuracy, Dict(x=>testx, y_=>testy)))
