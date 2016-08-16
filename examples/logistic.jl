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
gradW, gradB = gradients(Loss, [W, B])
Alpha = placeholder(Float64)
gradUpdate = [assign(W, W-Alpha.*gradW), assign(B, B-Alpha.*gradB)]

# Run training
run(sess, initialize_all_variables())

for epoch in 1:100
    alpha = .01/(1+epoch)
    cur_loss, _ = run(sess, vcat(Loss, gradUpdate), Dict(X=>x, Y_obs=>y, Alpha=>alpha))
    println(@sprintf("Current loss is %.2f.", cur_loss))
end
