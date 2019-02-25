using Statistics

abstract type Model
end

mutable struct Sequential <: Model
    attrs::Dict
end

mutable struct Dense
    weights::TensorHandle
    bias::TensorHandle
end

function Dense(in_size::Integer, out_size::Integer)
    layer = Dense(constant(randn(in_size, out_size)), constant(zeros(out_size)))
    return layer
end

struct ReluLayer
end

function forward(r::ReluLayer, x)
    nn.relu(x)
end

struct SGD
    lr::TensorHandle
end

SGD(;lr=1e-3)= SGD(convert(TensorHandle, lr))

function Sequential()
    d = Dict()
    d["trainable"] = Set()   
    d["layers"]  = []
    Sequential(d)
end


function add(m::Sequential, d::Dense)
    set_trainable(m, d.weights)
    set_trainable(m, d.bias)
    push!(m.attrs["layers"], d)
end

function add(m::Sequential, layer)
    push!(m.attrs["layers"], layer)
end

function forward(d::Dense, x)
    Ops.bias_add(x*d.weights, d.bias)
end

function mse(y, y_target)
    return mean((y .- y_target) .^ 2)
end

function set_trainable(m::Sequential, tensor)
    push!(m.attrs["trainable"], tensor)
end

function compile(m::Sequential; optimizer=nothing, loss=nothing)
    m.attrs["optimizer"] = optimizer
    m.attrs["loss"] = loss
end

function optimizier_step(g::SGD, value, grads)
    inplace_sub(value, g.lr .* grads)
end

function fit(m::Sequential, x, y; n_epochs=1, batch_size=nothing)
    optimizer = m.attrs["optimizer"]
    for epoch in 1:n_epochs
        tape = set_tape()
        y_predicted = x
        for layer in m.attrs["layers"]
            y_predicted = forward(layer, y_predicted)
        end
        loss = m.attrs["loss"](y, y_predicted)
        println("Epoch $epoch: Loss if $(item(loss))")
        values = collect(m.attrs["trainable"])
        grads = grad(tape, loss, values)
        for (value, g) in zip(values, grads)
            if g === nothing
                continue
            end
            optimizier_step(optimizer, value, g)
            # inplace_sub(value, lr.*g)
        end
    end
end
