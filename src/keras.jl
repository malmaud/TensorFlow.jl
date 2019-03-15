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
    layers::Vector{Layer}
    loss  # TODO constrain these fields more
    optimizer
    trainable::Set
end

@callable struct Dense <: Layer
    weights::EagerTensor
    bias::EagerTensor
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
    lr::EagerTensor
end

SGD(;lr=1e-3)= SGD(convert(EagerTensor, lr))

Sequential() = Sequential([], nothing, nothing, Set())

function add(m::Sequential, d::Dense)
    set_trainable(m, d.weights)
    set_trainable(m, d.bias)
    push!(m.layers, d)
end

add(m::Sequential, layer) = push!(m.layers, layer)

forward(d::Dense, x) = Ops.bias_add(x*d.weights, d.bias)

function forward(m::Sequential, x)
    for layer in m.layers
        x = forward(layer, x)
    end
    return x
end

mse(y, y_target) = mean((y .- y_target) .^ 2)

function set_trainable(m::Sequential, tensor)
    push!(m.trainable, tensor)
end

function compile(m::Sequential; optimizer=nothing, loss=nothing)
    m.optimizer = optimizer
    m.loss = loss
end

optimizier_step(g::SGD, value, grads) = inplace_sub(value, g.lr .* grads)

function fit(m::Sequential, x, y; n_epochs=1, batch_size=nothing)
    optimizer = m.optimizer
    for epoch in 1:n_epochs
        tape = create_tape()
        y_predicted = x
        for layer in m.layers
            y_predicted = forward(layer, y_predicted)
        end
        loss = m.loss(y, y_predicted)
        @info "" epoch loss=item(loss)
        values = collect(m.trainable)
        grads = grad(tape, loss, values)
        for (value, g) in zip(values, grads)
            if g === nothing
                continue
            end
            optimizier_step(optimizer, value, g)
        end
    end
end
