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

using Compat
using JLD2
using FileIO
using ProtoBuf
import Printf

import ..TensorFlow: Graph, Operation, get_def_graph, extend_graph, gradients, variable_scope, ConstantInitializer, node_name, get_variable, get_shape, get_collection, Session, placeholder, Tensor, Variable, cast, group, @not_implemented, AbstractQueue, tensorflow, add_to_collection, get_proto, get_def, @op

import TensorFlow
const tf = TensorFlow

abstract type Optimizer end

@op function minimize(optimizer::Optimizer, loss; global_step=nothing, var_list=nothing, name=nothing)
    if name == nothing
        name = optimizer.name
    end
    grads = compute_gradients(optimizer, loss, var_list)
    apply = apply_gradients(optimizer, grads; global_step=global_step, name=name)
end

function compute_gradients(optimizer::Optimizer, loss, var_list=nothing)
    if var_list === nothing
        var_list = get_def_graph().collections[:TrainableVariables]
    end
    vars = zip(gradients(loss, var_list), var_list) |> collect
    filter!(x->x[1]!==nothing, vars)  # Remove variables uninvolved in loss calculation
    vars
end

macro advance_step()
    quote
        if global_step !== nothing
            push!(ops, tf.assign(global_step, global_step+1))
        end
    end |> esc
end

mutable struct GradientDescentOptimizer <: Optimizer
    learning_rate::Tensor
    name::String
end

GradientDescentOptimizer(learning_rate; name="descent") = GradientDescentOptimizer(constant(learning_rate), name)

function GradientDescentOptimizer(; α=.01, kwargs...)
    GradientDescentOptimizer(α; kwargs...)
end

function Base.show(io::IO, optimizer::GradientDescentOptimizer)
    rate = optimizer.learning_rate
    print(io, "GradientDescentOptimizer(α=$rate)")
end

function general_assign_sub(var, learning_rate, grad::Tensor)
    tf.assign_sub(var, convert(Tensor{eltype(var)}, learning_rate) .* grad)
end

function general_assign_sub(var, learning_rate, grad::tf.IndexedSlices)
    tf.scatter_sub(var.var_node, grad.indices, convert(Tensor{eltype(var)}, learning_rate) .* grad.values)
end

function apply_gradients(optimizer::GradientDescentOptimizer, grads_and_vars; global_step=nothing, name="descent")
    ops = Tensor[]
    for (grad, var) in grads_and_vars
        push!(ops, general_assign_sub(var, optimizer.learning_rate, grad))
    end
    @advance_step
    return group(ops...)
end

mutable struct MomentumOptimizer <: Optimizer
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
        learning_rate = convert(Tensor{eltype(var)}, optimizer.learning_rate)
        momentum_rate = convert(Tensor{eltype(var)}, optimizer.momentum)
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

mutable struct AdamOptimizer <: Optimizer
    η::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    name::String
end

AdamOptimizer(learning_rate; β1=.9, β2=.999, ϵ=1e-8, name="adam") = AdamOptimizer(learning_rate, β1, β2, ϵ, name)

function AdamOptimizer(; η=.001, kwargs...)
    AdamOptimizer(η; kwargs...)
end

function Base.show(io::IO, optim::AdamOptimizer)
    print(io, "AdamOptimizer(η=$(optim.η), β1=$(optim.β1), β2=$(optim.β2), ϵ=$(optim.ϵ))")
end

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
        t = convert(Tensor{eltype(var)}, T)
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
            v_new = β2 .* v + (1-β2).*(grad.*grad)
            push!(ops, tf.assign_sub(var, lr/(sqrt(v_new)+ϵ) .* m_new))
            push!(ops, tf.assign(m, m_new))
            push!(ops, tf.assign(v, v_new))
        end
    end
    return group(ops...)
end

mutable struct Saver
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
        path = Printf.@sprintf("%s-%d", path, global_step)
    end
    jldopen(path, "w") do file
        for var_node in saver.var_list
            var_value = run(session, var_node)
            write(file, node_name(var_node)[1], var_value)
        end
    end
    versions = Int[]
	dirpath = dirname(path)
	if dirpath==""
		# default to current directory
		dirpath = "."
	end

    for file in readdir(dirpath)
        m = match(Regex("$(base_path)-(\\d+)"), file)
        if m !== nothing
            push!(versions, parse(Int, m[1]))
        end
    end
    if length(versions) > saver.max_to_keep
        to_delete = length(versions) - saver.max_to_keep
        sort!(versions)
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
function read_meta_graph_file(filepath::String)
    meta_graph_def = open(filepath) do f
        readproto(f, tensorflow.MetaGraphDef())
    end
    meta_graph_def
end

"""
    import_meta_graph

Recreates a Graph saved in a `MetaGraphDef` protocol buffer.

This function takes a `MetaGraphDef` protocol buffer as input. If the argument
is a file containing a `MetaGraphDef` protocol buffer, it constructs a protocol
buffer from the file content. The function then adds all the nodes from the
`graph_def` field to the current graph, recreates all the collections, and
returns a saver constructed from the `saver_def` field.

Assumes variables are trainable, unless the `trainable` keyword is provided, in
which case only variables whose names are in the list are "trainable".

Currently ignores all information under `save/*`. It also doesn't yet handle
QueueRunners and Summaries.
"""
function import_meta_graph(
        meta_graph_def::tensorflow.MetaGraphDef,
        graph::Graph;
        trainable::Vector{String} = String[]
    )

    nodes = meta_graph_def.graph_def.node
    extend_graph(graph, nodes)
    for node in nodes
        if tf.Variables.is_variable(node)
            domain = split(node.name, "/")[1]
            if domain !== "save"
                var_tensor = tf.Tensor(tf.get_node_by_name(graph, node.name), 1)
                assign_name = "$(node.name)/Assign"
                assign_tensor = tf.Tensor(tf.get_node_by_name(graph, assign_name), 1)
                var = Variable(var_tensor, assign_tensor)
                add_to_collection(graph, :Variables, var)
                if isempty(trainable) || (node.name in trainable)
                    add_to_collection(graph, :TrainableVariables, var)
                end
            end
        end
    end

    Saver(var_list = get_collection(graph, :TrainableVariables))
end

function import_meta_graph(
        meta_graph_def::tensorflow.MetaGraphDef;
        trainable::Vector{String} = String[]
    )
    import_meta_graph(
        meta_graph_def,
        get_def_graph(),
        trainable = trainable
    )
end

function import_meta_graph(
        filepath::String,
        graph::Graph;
        trainable::Vector{String} = String[]
    )
    import_meta_graph(
        read_meta_graph_file(filepath),
        graph,
        trainable = trainable
    )
end

function import_meta_graph(
        filepath::String,
        trainable::Vector{String} = String[]
    )
    import_meta_graph(
        read_meta_graph_file(filepath),
        get_def_graph(),
        trainable = trainable
    )
end

"Construct and returns a `MetaGraphDef` protocol buffer."
function create_meta_graph_def(graph::Graph)
    # reference: https://github.com/tensorflow/tensorflow/blob/cd1fe4072b43e650d47187838b7a37b050ec3d75/tensorflow/python/framework/meta_graph.py#L309-L385
    graph_def = get_def(graph)
    meta_info_def = tensorflow.MetaGraphDef_MetaInfoDef()
      # TODO: Set the tf version strings to the current tf build.
      # meta_graph_def.meta_info_def.tensorflow_version = versions.__version__
      # meta_graph_def.meta_info_def.tensorflow_git_version = versions.__git_version__
    collection_def = Dict{String,tensorflow.CollectionDef}()
    # In general, we should provide a registration mechanism for `to_proto`
    # functions to serialize objects into their corresponding protocol buffers.
    # (see https://github.com/tensorflow/tensorflow/blob/799e31f3840c21322e380e1ec6e5bacb95d016fa/tensorflow/python/framework/ops.py#L4277-L4300)
    # (and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/meta_graph.py#L274-L282)
    for (name, collection) in graph.collections
        bytes_list = Array{UInt8,1}[]
        for obj in collection
            # TODO: complete it for the other types
            if isa(obj, Variable)
                push!(bytes_list, get_proto(obj.var_node))
                push!(bytes_list, get_proto(obj.assign_node))
            elseif isa(obj, Tensor)
                push!(bytes_list, get_proto(obj.op))
            elseif isa(obj, tensorflow.WhileContextDef)
                push!(bytes_list, get_proto(obj))
            end
        end
        collectiondef = tensorflow.CollectionDef(;bytes_list=tensorflow.CollectionDef_BytesList(;value=bytes_list))
        collection_def[string(name)] = collectiondef
    end
    tensorflow.MetaGraphDef(; graph_def=graph_def, meta_info_def=meta_info_def, collection_def=collection_def)
end
create_meta_graph_def() = create_meta_graph_def(get_def_graph())

"""
Writes a MetaGraphDef to filepath.

The exported CollectionDef currently only contains operations from `Variables`
and `Tensors`.
"""
function export_meta_graph(filepath::String="")
    # Although we don't use it yet, the function signature has ::Saver to match
    # the corresponding python function.
    meta_graph = create_meta_graph_def()
    if !isempty(filepath)
        open(filepath, "w") do f
            writeproto(f, create_meta_graph_def())
        end
    end
    meta_graph
end

include("train/pipeline.jl")

## Deprecations

function SummaryWriter(args...; kwargs...)
    Base.depwarn("train.SummaryWriter is deprecated. Use summary.FileWriter instead.", :SummaryWriter)
    TensorFlow.summary.FileWriter(args...; kwargs...)
end

end
