using Distributions

type Variable <: AbstractTensor
    var_node::Operation
    assign_node::Operation

    Variable() = new()
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
    desc = NodeDescription("Variable", name)
    desc["dtype"] = eltype(initial_value)
    desc["shape"] = size(initial_value)
    self.var_node = Operation(desc)

    desc = NodeDescription("Assign", "$name/Assign")
    add_input(desc, self.var_node)
    t = Tensor(initial_value)
    add_input(desc, t)
    self.assign_node = Operation(desc)
    add_to_collection(:Variables, self)
    if trainable
        add_to_collection(:TrainableVariables, self)
    end
    return self
end

function assign(v::Variable, value)
    desc = NodeDescription(get_def_graph(), "Assign", get_name())
    add_input(desc, v.var_node)
    add_input(desc, Tensor(value))
    return Tensor(Operation(desc), 1)
end

function assign_sub(v::Variable, value)
    desc = NodeDescription("AssignSub", get_name())
    add_input(desc, v.var_node)
    add_input(desc, Tensor(value))
    return Tensor(Operation(desc), 1)
end

function scatter_update(ref, indices, updates; name="ScatterUpdate")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ScatterUpdate")
        add_input(desc, Tensor(ref))
        add_input(desc, Tensor(indices)-1)
        add_input(desc, Tensor(updates))
    end
    Tensor(Operation(desc))
end

function scatter_sub(ref, indices, updates; name="ScatterSub")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ScatterSub")
        add_input(desc, Tensor(ref))
        add_input(desc, Tensor(indices)-1)
        add_input(desc, Tensor(updates))
    end
    Tensor(Operation(desc))
end

Base.setindex!(v::Variable, value) = assign(v, value)

Base.convert(::Type{Tensor}, v::Variable) = Tensor(v.var_node, 1)

"""
Returns an `Operation` that initializes all TensorFlow `Variable`s.
"""
function initialize_all_variables()
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

get_dims(t::AbstractTensorShape) = map(get, t.dims)
get_dims(x) = x

"""
Gets an existing variable with these parameters (`shape`, `dtype`, `trainable`)
or create a new one.
"""
function get_variable(var_name, shape, dtype; trainable=true, kwargs...)
    shape = get_dims(shape)
    scope = make_scope(var_name; kwargs...)
    push!(scope_stack, scope)
    name = join([get(_.name) for _ in scope_stack], "/")
    local v
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
            n = get_node_by_name(get_def_graph(), name)
            v = Variable()
            v.var_node = get_node_by_name(name) |> get
            v.assign_node = get_node_by_name("$name/Assign") |> get
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
    return v
end

type ConstantInitializer{T}
    value::T
end

function Base.rand(c::ConstantInitializer, shape...)
    fill(c.value, shape)
end
