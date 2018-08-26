#=
An example implementaion of an Autoencoder inspired by
https://github.com/alrojo/tensorflow-tutorial/blob/master/lab5_AE/lab5_AE.ipynb
This example trains an undercomplete autoencoder with a bottleneck of three hidden neurons, the activation of which can be used for a low-dimensional visualization of the 768-dimensional MNIST data.

The plotting functionality require Plots.jl and a backend, e.g., PyPlot.jl or GR.jl
`Pkg.add("Plots")`
`Pkg.add("PyPlot")` or `Pkg.add("GR")`
=#
using MNIST, Plots, TensorFlow, ValueHistories

function showdigit(x; kwargs...)
    d = reshape(x,28,28)
    heatmap(d; yflip=true, color=:grays, kwargs...)
end

function printtime(t0)
    dt = time()-t0
    "$(Int(dt÷60)):$(round(Int,dt % 60))"
end


num_features = 28^2


include(joinpath(dirname(pathof(TensorFlow)), "..", "examples","mnist_loader.jl"))
include(joinpath(dirname(pathof(TensorFlow)), "..", "src","layers","fully_connected.jl"))
loader = DataLoader()
session = Session(Graph())

x = placeholder(Float32, shape=[-1, num_features], name="x_pl")#/255
variable_scope("ae", initializer=Distributions.Normal(0, .01)) do
    global W1 = get_variable("weights1", [num_features, 512], Float32)
    global W2 = get_variable("weights2", [512, 3], Float32)
    global W1o = get_variable("weights1o", [512, num_features], Float32)
    global W2o = get_variable("weights2o", [3, 512], Float32)
end


B1 = Variable(0.01ones(Float32,512), name="bias1")
B2 = Variable(0.01ones(Float32,3), name="bias2")
B1o = Variable(0.01ones(Float32,num_features), name="bias1o")
B2o = Variable(0.01ones(Float32,512), name="bias2o")

# These lines define the network structure
l_enc = x*W1 + B1 |> nn.relu
l_z = l_enc*W2 + B2
l_dec = l_z*W2o + B2o |> nn.relu
l_out = l_dec*W1o + B1o |> nn.sigmoid

# loss_CE = -reduce_sum(x.*log(l_out+1e-8) + (1-x).*log(1-l_out+1e-8))
loss_MSE = reduce_mean((x-l_out).^2)
train_step = train.minimize(train.AdamOptimizer(5e-4), loss_MSE)

run(session, global_variables_initializer())
saver = train.Saver()
checkpoint_path = mktempdir()

summary_writer = train.SummaryWriter(checkpoint_path)


testx, testy = load_test_set()
testx ./= 255
testy = sum(testy.*(0:9)',2)[:]

f = plot(legend=false)
history = MVHistory()
@time begin
    t0 = time()
    for epoch = 1:10_000
        batch,_ = next_batch(loader, 64)
        batch ./= 255 # Normalize input to [0,1]
        if epoch%500 == 0
            train_loss = run(session, loss_MSE, Dict(x=>batch))
            val_loss, center = run(session, [loss_MSE, l_z], Dict(x=>testx))
            push!(history,:loss_train, epoch, train_loss)
            push!(history,:loss_val, epoch, val_loss)
            plot(history, reuse=true)
            scatter3d(center[:,1],center[:,2],center[:,3], zcolor = testy, legend=false, title="Latent space", reuse=true)
            @info("step $epoch, training loss $train_loss, time taken: $(printtime(t0))")
            train.save(saver, session, joinpath(checkpoint_path, "ae_mnist"), global_step=epoch)
        end
        run(session, train_step, Dict(x=>batch))
    end
end

test_loss, center, reconstruction = run(session, [loss_MSE, l_z, l_out], Dict(x=>testx))
@info("test accuracy $test_loss")

# Plot som example reconstructions
offset = 0
plot(layout=(2,2*2))
for i = 1:4
    d = reshape(testx[i+offset,:],28,28)
    heatmap!(d, subplot=i,yflip=true, color=:grays, title="Input", colorbar = false)
    d = reshape(reconstruction[i+offset,:],28,28)
    heatmap!(d, subplot=i+4,yflip=true, color=:grays, title="Reconstruction", colorbar = false)
end
