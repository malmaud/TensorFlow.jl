using TensorFlow
using Distributions
using Printf

# Generate some synthetic data
x = randn(100, 50)
w = randn(50, 10)
y_prob = exp.(x*w)
y_prob ./= sum(y_prob,dims=2)

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

X = placeholder(Float64, shape=[-1, 50])
Y_obs = placeholder(Float64, shape=[-1, 10])

variable_scope("logisitic_model"; initializer=Normal(0, .001)) do
    global W = get_variable("W", [50, 10], Float64)
    global B = get_variable("B", [10], Float64)
end

Y=nn.softmax(X*W + B)

Loss = -reduce_sum(log(Y).*Y_obs)
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)
saver = train.Saver()

# Run training
run(sess, global_variables_initializer())
checkpoint_path = mktempdir()
@info("Checkpoint files saved in $checkpoint_path")
for epoch in 1:100
    cur_loss, _ = run(sess, [Loss, minimize_op], Dict(X=>x, Y_obs=>y))
    println(@sprintf("Current loss is %.2f.", cur_loss))
    train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)
end
