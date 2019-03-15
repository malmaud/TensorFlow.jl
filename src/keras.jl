using Statistics


abstract type KerasCallable
end

abstract type Model <: KerasCallable
end

abstract type Layer <: KerasCallable
end

function struct_name(f)
    @capture(f, struct name_ <: _
        __
    end) && return name
    @capture(f, mutable struct name_ <: _
        __
    end) && return name
    return nothing
end

# Get around https://github.com/JuliaLang/julia/issues/14919
macro callable(f)
    name = struct_name(f)
    quote
        $(esc(f))
        (m::$name)(args...; kwargs...) = forward(m, args...; kwargs...)
    end
end

@callable mutable struct Sequential <: Model
    attrs::Dict
end

@callable mutable struct Dense <: Layer
    weights::TensorHandle
    bias::TensorHandle
end

function Dense(in_size::Integer, out_size::Integer)
    layer = Dense(constant(randn(in_size, out_size)), constant(zeros(out_size)))
    return layer
end

@callable struct Relu <: Layer
end

function forward(r::Relu, x)
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

forward(d::Dense, x) = Ops.bias_add(x*d.weights, d.bias)

function forward(m::Sequential, x)
    for layer in m.attrs["layers"]
        x = forward(layer, x)
    end
    return x
end

mse(y, y_target) = mean((y .- y_target) .^ 2)

function set_trainable(m::Sequential, tensor)
    push!(m.attrs["trainable"], tensor)
end

function compile(m::Sequential; optimizer=nothing, loss=nothing)
    m.attrs["optimizer"] = optimizer
    m.attrs["loss"] = loss
end

optimizier_step(g::SGD, value, grads) = inplace_sub(value, g.lr .* grads)

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
        end
    end
end
