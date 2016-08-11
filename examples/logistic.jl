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
sess = Session()
X = placeholder(Float64)
Y_obs = placeholder(Float64)
W = Variable(randn(50,10))
Y=nn.softmax(X*W)
Loss = -reduce_sum(log(Y).*Y_obs)
grad = gradients(Loss, W)

# Run training
run(sess, initialize_all_variables())

for epoch in 1:100
    cur_grad,cur_loss=run(sess, [grad,Loss], Dict(X=>x, Y_obs=>y))
    println(@sprintf("Current loss is %.2f. Gradient norm is %.2f", cur_loss, sum(cur_grad)))
    run(sess, assign(W, W-.0001*cur_grad))
end
