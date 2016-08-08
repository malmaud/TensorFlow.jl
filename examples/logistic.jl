using Distributions

x = randn(100, 3)

w = randn(3, 10)

y = x*w

sess = Session()
constant(1)
run(sess, constant(1))

X = placeholder(Float64)

Y_obs = placeholder(Float64)

W = Variable(randn(3,10))
Y=X*W

Loss = reduce_sum((Y-Y_obs)^2)
run(sess, initialize_all_variables())
run(sess, Loss, Dict(X=>x, Y_obs=>y))

grad = gradients(Loss, W)

run(sess, grad, Dict(X=>x, Y_obs=>y))
