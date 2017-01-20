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
restore,
SummaryWriter,
add_queue_runner,
start_queue_runners,
clear_queue_runners,
range_input_producer,
input_producer,
string_input_producer,
shuffle_batch,
QueueRunner,
create_threads

using JLD
using FileIO
using ProtoBuf

import ..TensorFlow: Operation, get_def_graph, gradients, variable_scope, ConstantInitializer, node_name, get_variable, get_shape, get_collection, Session, placeholder, Tensor, cast, group, @not_implemented, AbstractQueue, tensorflow

import TensorFlow
const tf = TensorFlow

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
            push!(ops, tf.assign(global_step, global_step+1))
        end
    end
end

type GradientDescentOptimizer <: Optimizer
    learning_rate::Tensor
    name::String
end

GradientDescentOptimizer(learning_rate; name="descent") = GradientDescentOptimizer(Tensor(learning_rate), name)

function general_assign_sub(var, learning_rate, grad::Tensor)
    tf.assign_sub(var, cast(learning_rate, eltype(var)) .* grad)
end

function general_assign_sub(var, learning_rate, grad::tf.IndexedSlices)
    tf.scatter_sub(var.var_node, grad.indices, cast(learning_rate, eltype(var)) .* grad.values)
end

function apply_gradients(optimizer::GradientDescentOptimizer, grads_and_vars; global_step=nothing, name="descent")
    ops = Tensor[]
    for (grad, var) in grads_and_vars
        push!(ops, general_assign_sub(var, optimizer.learning_rate, grad))
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
            variable_scope(node_name(var)[1]) do
                momentum = get_variable("momentum", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0), trainable=false)
            end
        end
        learning_rate = cast(optimizer.learning_rate, eltype(var))
        momentum_rate = cast(optimizer.momentum, eltype(var))
        if isa(grad, tf.IndexedSlices)
            momentum_slice = tf.gather(momentum, grad.indices) # TODO should reduce?
            step = learning_rate .* grad.values + momentum_rate .* momentum_slice
            push!(ops, tf.scatter_sub(var.var_node, grad.indices, step))
            push!(ops, tf.scatter_update(momentum, grad.indices, step))
        else
            step = learning_rate .* grad + momentum_rate .* momentum
            push!(ops, tf.assign(var, var-step))  # Problematic line
            push!(ops, tf.assign(momentum, step))
        end

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

AdamOptimizer(learning_rate=.001; β1=.9, β2=.999, ϵ=1e-8, name="adam") = AdamOptimizer(learning_rate, β1, β2, ϵ, name)

function apply_gradients(optimizer::AdamOptimizer, grads_and_vars; global_step=nothing, name="adam")
    ops = Tensor[]
    @advance_step
    for (grad, var) in grads_and_vars
        local m, v, T
        variable_scope(name) do
            variable_scope(node_name(var)[1]) do
                m = get_variable("m", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0), trainable=false)
                v = get_variable("v", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0), trainable=false)
                T = get_variable("t", [], Float32, initializer=ConstantInitializer(1.0), trainable=false)
            end
        end
        β1 = eltype(var)(optimizer.β1)
        β2 = eltype(var)(optimizer.β2)
        ϵ = eltype(var)(optimizer.ϵ)
        η = eltype(var)(optimizer.η)
        t = cast(Tensor(T), eltype(var))
        push!(ops, tf.assign(T, T+1))
        lr = η*sqrt(1-β2^t)/(1-β1^t)
        if isa(grad, tf.IndexedSlices)
            m_slice = tf.gather(m, grad.indices)
            v_slice = tf.gather(v, grad.indices)
            m_new = β1 .* m_slice + (1-β1) .* grad.values
            v_new = β2 .* v_slice + (1-β2) .* (grad.values .^ 2)
            push!(ops, tf.scatter_sub(var.var_node, grad.indices, lr/(sqrt(v_new)+ϵ) .* m_new))
            push!(ops, tf.scatter_update(m.var_node, grad.indices, m_new))
            push!(ops, tf.scatter_update(v.var_node, grad.indices, v_new))
        else
            m_new = β1 .* m + (1-β1).*grad
            v_new = β2 .* v + (1-β2).*(grad.^2)
            push!(ops, tf.assign_sub(var, lr/(sqrt(v_new)+ϵ) .* m_new))
            push!(ops, tf.assign(m, m_new))
            push!(ops, tf.assign(v, v_new))
        end
    end
    return group(ops...)
end

type Saver
    var_list
    max_to_keep
    placeholder_nodes
    restore_ops
end

function Base.show(io::IO, saver::Saver)
    print(io, "<Saver>")
end

function Saver(;var_list=nothing, max_to_keep=5)
    if var_list === nothing
        var_list = get_collection(:TrainableVariables)
    end
    placeholders = Dict()
    restore_ops = []
    for var in var_list
        ph = placeholder(eltype(var))
        placeholders[node_name(var)[1]] = ph
        restore_op = tf.assign(var, ph)
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
            write(file, node_name(var_node)[1], var_value)
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

function restore_helper!(feed_dict::Dict, saver::Saver, prefix::String, namespace_dict::Dict)
    for (var_name, var_value) in namespace_dict
        restore_helper!(feed_dict, saver, prefix*"/"*var_name, var_value)
    end
end

function restore_helper!(feed_dict::Dict, saver::Saver, var_name::String, var_value)
    placeholder = saver.placeholder_nodes[var_name]
    feed_dict[placeholder] = var_value
end

function restore(saver::Saver, session::Session, save_path)
    d = Dict()
    checkpoint = load(save_path)
    for (var_name, var_value) in checkpoint
        restore_helper!(d, saver, var_name, var_value)
    end
    run(session, saver.restore_ops, d)
    nothing
end

"""
Reads a file containing `MetaGraphDef` and returns the protocol buffer.

Args:
    filepath: `meta_graph_def` filepath

Returns:
    A `MetaGraphDef` protocol buffer.
"""
function read_meta_graph_file(filepath)
    meta_graph_def = open(filepath) do f
        readproto(f, tensorflow.MetaGraphDef())
    end
    meta_graph_def
end

# TODO: implement
#  (i) import_scoped_meta_graph (https://github.com/tensorflow/tensorflow/blob/99fe61a8a8f3dd41b4e1e4dedfc53b45f67e88a7/tensorflow/python/framework/meta_graph.py#L420-L544)
#  (ii) export_scoped_meta_graph (https://github.com/tensorflow/tensorflow/blob/99fe61a8a8f3dd41b4e1e4dedfc53b45f67e88a7/tensorflow/python/framework/meta_graph.py#L547-L649)

include("train/summary_writer.jl")
include("train/pipeline.jl")

end
