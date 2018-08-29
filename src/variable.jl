module Variables

export
assign,
assign_add,
assign_sub,
scatter_update,
scatter_sub,
scatter_add,
scatter_mul,
scatter_div,
Variable,
variable_scope,
get_variable,
global_variables_initializer

import TensorFlow
const tf = TensorFlow

import .tf.Ops:
    assign,
    assign_add,
    assign_sub,
    scatter_update,
    scatter_sub,
    scatter_add,
    scatter_mul,
    scatter_div

import Distributions

mutable struct Variable{T} <: tf.AbstractTensor{T}
    var_node::tf.Tensor{T}
    assign_node::tf.Tensor{T}

    Variable{T}() where {T} = new{T}()
end

function Variable(var_node::tf.Tensor{T}, assign_node::tf.Tensor{T}) where T
    v = Variable{T}()
    v.var_node = var_node
    v.assign_node = assign_node
    v
end

"""
A variable maintains state in the graph across calls to `run()`.
You add a variable to the graph by constructing an instance of the type `Variable`.

The `Variable()` constructor requires an `initial_value` for the variable,
which can be a `Tensor` of any type and shape. The initial value defines the
type and shape of the variable. After construction, the type and shape of the
variable are fixed. The value can be changed using one of the `assign` methods.

If you want to change the shape of a variable later you have to use an
`assign` `Operation` with `validate_shape=false`.

Variables can be used as inputs to `Operation`s, just like any `Tensor`.

When you launch the graph, variables have to be explicitly initialized before
you can run `Operation`s that use their value. You can initialize a variable by
running its `initializer` `Operation`, restoring the variable from a save file,
or simply running an `assign` `Operation` that assigns a value to the variable.
"""
function Variable(initial_value; name="", trainable=true, literal_name=false)
    self = Variable{eltype(initial_value)}()
    if !literal_name
        name = tf.get_name(name)
    end
    self.var_node = tf.Ops.variable_v2(name=name, dtype=eltype(initial_value), shape=tf.TensorShape([size(initial_value)...]))
    self.assign_node = tf.Ops.assign(tf.Tensor(self.var_node), initial_value, name="$name/Assign")
    tf.add_to_collection(:Variables, self)
    if trainable
        tf.add_to_collection(:TrainableVariables, self)
    end
    return self
end

@tf.with_def_graph function Variable(graph::tf.Graph, s::AbstractString)
    var_node = tf.Tensor(tf.get_node_by_name(graph, s))
    assign_node = tf.Tensor(tf.get_node_by_name(graph, "$s/Assign"))
    Variable(var_node, assign_node)
end

Base.setindex!(v::Variable, value) = assign(v, value)

Base.convert(::Type{tf.Tensor}, v::Variable) = v.var_node

"""
Returns an `Operation` that initializes all TensorFlow `Variable`s.
"""
function global_variables_initializer()
    return tf.group([var.assign_node for var in tf.get_collection(:Variables)]...)
end

run(sess::tf.Session, var::Variable) = run(sess, tf.Tensor(var))
run(sess::tf.Session, vars::AbstractVector{Variable}) = run(sess, map(tf.Tensor, vars))

mutable struct Scope
    name::Union{String, Nothing}
    initializer::Any
    reuse::Bool
    Scope() = new(nothing, nothing, false)
end

const scope_stack = Scope[]

function make_scope(name; initializer=nothing, reuse=false)
    scope = Scope()
    scope.name = name
    if initializer != nothing
        scope.initializer = initializer
    end
    scope.reuse = reuse
    return scope
end

"""
Returns a context manager for defining `Operation`s that create `Variable`s (layers).

This context manager validates that the (optional) values are from the same graph,
ensures that graph is the default graph, and pushes a name scope and a variable scope.
Variable scope allows one to create new variables and to share already created
ones while providing checks to not create or share variables by accident.
"""
function variable_scope(f, name; kwargs...)
    scope = make_scope(name; kwargs...)
    push!(scope_stack, scope)
    try
        f()
    finally
        pop!(scope_stack)
    end
end

get_dims(t::tf.TensorShape) = t.dims
get_dims(x) = x

struct NormalInitializer
    sd::Float64
end

NormalInitializer() = NormalInitializer(.01)

Base.rand(rng::NormalInitializer, shape::Integer...) = rng.sd * randn(shape)

"""
Gets an existing variable with these parameters (`shape`, `dtype`, `trainable`)
or create a new one.
"""
function tf.get_variable(var_name, shape, dtype; trainable=true, kwargs...)
    local v
    tf.with_top_level() do
        shape = get_dims(shape)
        scope = make_scope(var_name; kwargs...)
        push!(scope_stack, scope)
        name = join([x.name for x in scope_stack], "/")
        try
            initializer = NormalInitializer()
            reuse = false
            for scope in scope_stack
                if scope.initializer !== nothing
                    initializer = scope.initializer
                end
                if scope.reuse
                    reuse = true
                end
            end
            if reuse
                v = Variable(name)
            else
                if length(shape) > 0
                    iv = rand(initializer, shape...)
                else
                    iv = rand(initializer, 1)[1]
                end
                v = Variable(map(dtype, iv), name=name, trainable=trainable, literal_name=true)
            end
        finally
            pop!(scope_stack)
        end
    end
    return v
end

function tf.get_variable(var_name; shape=nothing, dtype=nothing, kwargs...)
    @tf.required shape dtype
    tf.get_variable(var_name, shape, dtype; kwargs...)
end

tf.get_tensors(v::Variable) = [v.var_node]

function is_variable(name::AbstractString)
    return occursin(r"^(Variable|VariableV\d+)$", name)
end

is_variable(name::Union{tf.Operation, tf.AbstractTensor}) = is_variable(tf.get_op(name).op_name)

is_variable(def::tf.tensorflow.NodeDef) = is_variable(def.op)

end
