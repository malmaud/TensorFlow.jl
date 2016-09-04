module train

export
minimize,
compute_gradients,
apply_gradients,
GradientDescentOptimizer,
MomentumOptimizer,
AdamOptimizer,
Saver,
save,
restore

using JLD
using FileIO

import ..TensorFlow: Operation, get_def_graph, gradients, assign, variable_scope, ConstantInitializer, node_name, get_variable, get_shape, get_collection, Session, placeholder, Tensor, cast, group

abstract Optimizer

function minimize(optimizer::Optimizer, loss; global_step=nothing, var_list=nothing, name="")
    if name == ""
        name = optimizer.name
    end
    grads = compute_gradients(optimizer, loss, var_list)
    apply = apply_gradients(optimizer, grads; global_step=global_step, name=name)
end

function compute_gradients(optimizer::Optimizer, loss, var_list=nothing)
    if var_list === nothing
        var_list = get_def_graph().collections[:TrainableVariables]
    end
    zip(gradients(loss, var_list), var_list) |> collect
end

macro advance_step()
    quote
        if global_step !== nothing
            push!(ops, assign(global_step, global_step+1))
        end
    end
end

type GradientDescentOptimizer <: Optimizer
    learning_rate::Tensor
    name::String
end

GradientDescentOptimizer(learning_rate; name="descent") = GradientDescentOptimizer(Tensor(learning_rate), name)

function apply_gradients(optimizer::GradientDescentOptimizer, grads_and_vars; global_step=nothing, name="descent")
    ops = Tensor[]
    for (grad, var) in grads_and_vars
        push!(ops, assign(var, var - cast(optimizer.learning_rate, eltype(var)).*grad))
    end
    @advance_step
    return group(ops...)
end

type MomentumOptimizer <: Optimizer
    learning_rate::Tensor
    momentum::Tensor
    name::String
end

MomentumOptimizer(learning_rate, momentum; name="momentum") = MomentumOptimizer(learning_rate, momentum, name)

function apply_gradients(optimizer::MomentumOptimizer, grads_and_vars; global_step=nothing, name="momentum")
    ops = Tensor[]
    @advance_step
    for (grad, var) in grads_and_vars
        local momentum
        variable_scope(name) do
            variable_scope(node_name(var)) do
                momentum = get_variable("momentum", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0), trainable=false)
            end
        end
        step = -optimizer.learning_rate .* grad + optimizer.momentum .* momentum
        push!(ops, assign(var, var + step))
        push!(ops, assign(momentum, step))
    end
    return group(ops...)
end

type AdamOptimizer <: Optimizer
    η::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    name::String
end

AdamOptimizer(learning_rate=.01; name="adam") = AdamOptimizer(learning_rate, .9, .999, 1e-8, name)

function apply_gradients(optimizer::AdamOptimizer, grads_and_vars; global_step=nothing, name="adam")
    ops = Tensor[]
    @advance_step
    for (grad, var) in grads_and_vars
        local m, v
        variable_scope(name) do
            variable_scope(node_name(var)) do
                m = get_variable("m", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0), trainable=false)
                v = get_variable("v", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0), trainable=false)
            end
        end
        β1 = eltype(var)(optimizer.β1)
        β2 = eltype(var)(optimizer.β2)
        ϵ = eltype(var)(optimizer.ϵ)
        η = eltype(var)(optimizer.η)
        m_new = β1 .* m + (1-β1).*grad
        v_new = β2 .* v + (1-β2).*(grad.^2)
        # TODO use m_hat
        push!(ops, assign(var, var - η/(sqrt(v_new)+ϵ) .* m_new))
        push!(ops, assign(m, m_new))
        push!(ops, assign(v, v_new))
    end
    return group(ops...)
end

type Saver
    var_list
    max_to_keep
    placeholder_nodes
    restore_ops
end

function Saver(;var_list=nothing, max_to_keep=5)
    if var_list === nothing
        var_list = get_collection(:TrainableVariables)
    end
    placeholders = Dict()
    restore_ops = []
    for var in var_list
        ph = placeholder(eltype(var))
        placeholders[node_name(var)] = ph
        restore_op = assign(var, ph)
        push!(restore_ops, restore_op)
    end
    Saver(var_list, max_to_keep, placeholders, restore_ops)
end

function FileIO.save(saver::Saver, session::Session, path; global_step=nothing)
    base_path = basename(path)
    if global_step !== nothing
        path = @sprintf("%s-%d", path, global_step)
    end
    jldopen(path, "w") do file
        for var_node in saver.var_list
            var_value = run(session, var_node)
            write(file, node_name(var_node), var_value)
        end
    end
    versions = Int[]
    for file in readdir(dirname(path))
        m = match(Regex("$(base_path)-(\\d+)"), file)
        if m !== nothing
            push!(versions, parse(Int, m[1]))
        end
    end
    if length(versions) > saver.max_to_keep
        to_delete = length(versions) - saver.max_to_keep
        for i in 1:to_delete
            rm(joinpath(dirname(path), "$base_path-$(versions[i])"), force=true)
        end
    end
end

function restore(saver::Saver, session::Session, save_path)
    d = Dict()
    checkpoint = load(save_path)
    for (var_name, var_value) in checkpoint
        placeholder = saver.placeholder_nodes[var_name]
        d[placeholder] = var_value
    end
    run(session, saver.restore_ops, d)
end

include("summary_writer.jl")

end
