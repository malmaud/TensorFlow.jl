using Statistics

mutable struct Model
    attrs::Dict
end

mutable struct Dense
    weights
    bias
end

function dense(in_size, out_size)
    layer = Dense(constant(randn(in_size, out_size)), constant(zeros(out_size)))
    return layer
end

function Model()
    d = Dict()
    d["trainable"] = Set()   
    d["layers"]  = []
    Model(d)
end


function add(m::Model, d::Dense)
    set_trainable(m, d.weights)
    set_trainable(m, d.bias)
    push!(m.attrs["layers"], d)
end

function forward(d::Dense, x)
    x*d.weights #+ d.bias
end

function mse(y, y_target)
    return mean((y .- y_target) .^ 2)
end

function set_trainable(m::Model, tensor)
    push!(m.attrs["trainable"], tensor)
end

function compile(m::Model; optimizer=nothing, loss=nothing)
    m.attrs["optimizer"] = optimizer
    m.attrs["loss"] = loss
end

function fit(m::Model, x, y; n_epochs=1, batch_size=nothing)
    lr = constant(m.attrs["optimizer"])
    for epoch in 1:n_epochs
        tape = set_tape()
        y_predicted = forward(m.attrs["layers"][1], x)
        loss = m.attrs["loss"](y, y_predicted)
        println("Epoch $epoch: Loss if $(item(loss))")
        values = collect(m.attrs["trainable"])
        grads = grad(tape, loss, values)
        for (value, g) in zip(values, grads)
            if g === nothing
                continue
            end
            inplace_sub(value, lr.*g)
        end
    end
end
