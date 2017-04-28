import .Ops:
    assign,
    assign_add,
    assign_sub,
    scatter_update,
    scatter_sub,
    scatter_add,
    scatter_mul,
    scatter_div

import Distributions

type Variable <: AbstractTensor
    var_node::Operation
    assign_node::Operation

    Variable() = new()
    Variable(var_node::Operation, assign_node::Operation) =
        new(var_node, assign_node)
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
    self = Variable()
    if !literal_name
        name = get_name(name)
    end
    self.var_node = get_op(Ops.variable_v2(name=name, dtype=eltype(initial_value), shape=TensorShape([size(initial_value)...])))

    self.assign_node = get_op(Ops.assign(Tensor(self.var_node), initial_value, name="$name/Assign"))
    add_to_collection(:Variables, self)
    if trainable
        add_to_collection(:TrainableVariables, self)
    end
    return self
end

@with_def_graph function Variable(graph::Graph, s::AbstractString)
    var = Variable()
    var.var_node = get(get_node_by_name(graph, s))
    var.assign_node = get(get_node_by_name(graph, "$s/Assign"))
    var
end

Base.setindex!(v::Variable, value) = assign(v, value)

Base.convert(::Type{Tensor}, v::Variable) = Tensor(v.var_node, 1)

"""
Returns an `Operation` that initializes all TensorFlow `Variable`s.
"""
function global_variables_initializer()
    return group([Tensor(var.assign_node) for var in get_collection(:Variables)]...)
end

run(sess::Session, var::Variable) = run(sess, Tensor(var))
run(sess::Session, vars::AbstractVector{Variable}) = run(sess, map(Tensor, vars))

type Scope
    name::Nullable{String}
    initializer::Nullable{Any}
    reuse::Bool
    Scope() = new(Nullable{String}(), Nullable{Any}(), false)
end

const scope_stack = Scope[]

function make_scope(name; initializer=nothing, reuse=false)
    scope = Scope()
    scope.name = Nullable(name)
    if initializer != nothing
        scope.initializer = Nullable(initializer)
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

get_dims(t::TensorShape) = map(get, t.dims)
get_dims(x) = x

"""
Gets an existing variable with these parameters (`shape`, `dtype`, `trainable`)
or create a new one.
"""
function get_variable(var_name, shape, dtype; trainable=true, kwargs...)
    local v
    with_top_level() do
        shape = get_dims(shape)
        scope = make_scope(var_name; kwargs...)
        push!(scope_stack, scope)
        name = join([get(x.name) for x in scope_stack], "/")
        try
            initializer = Normal(0, .01)
            reuse = false
            for scope in scope_stack
                if !isnull(scope.initializer)
                    initializer = get(scope.initializer)
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
