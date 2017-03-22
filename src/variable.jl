using Distributions

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
    desc = NodeDescription("VariableV2", name)
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

@with_def_graph function Variable(graph::Graph, s::AbstractString)
    var = Variable()
    var.var_node = get(get_node_by_name(graph, s))
    var.assign_node = get(get_node_by_name(graph, "$s/Assign"))
    var
end

"""
Update `v` by assigning `value` to it.

Args:
* `v`: The `Variable` to update.
* `value`: The new value contained by `v`.
* `validate_shape`: Optional `Bool` which, if `true` (default), ensures that `v` and `value` have the same shape.
* `use_locking`: Optional `Bool` which, if `true` (default), protects the assignment with a lock.

Returns:
`v`, the updated `Variable`.
"""
@op function assign(v::Variable, value; validate_shape=true, use_locking=true, name=nothing)
    local desc
    with_op_name(name, "Assign") do
        desc = NodeDescription("Assign")
        add_input(desc, v.var_node)
        add_input(desc, Tensor(value))
        desc["validate_shape"] = validate_shape
        desc["use_locking"] = use_locking
    end
    return Tensor(Operation(desc), 1)
end

"""
Update `v` by adding `value` to it.

Args:
* `v`: The `Variable` to update.
* `value`: The new value to add to `v`.
* `use_locking`: Optional `Bool` which, if `true` (default is `false`), protects the assignment with a lock.

Returns:
`v`, the updated `Variable`.
"""
@op function assign_add(v::Variable, value; use_locking=false, name=nothing)
    local desc
    with_op_name(name, "AssignAdd") do
        desc = NodeDescription("AssignAdd")
        add_input(desc, v.var_node)
        add_input(desc, Tensor(value))
        desc["use_locking"] = use_locking
    end
    return Tensor(Operation(desc), 1)
end

"""
Update `v` by subtracting `value` from it.

Args:
* `v`: The `Variable` to update.
* `value`: The new value to subtract from `v`.
* `use_locking`: Optional `Bool` which, if `true` (default is `false`), protects the assignment with a lock.

Returns:
`v`, the updated `Variable`.
"""
@op function assign_sub(v::Variable, value; use_locking=false, name=nothing)
    local desc
    with_op_name(name, "AssignSub") do
        desc = NodeDescription("AssignSub")
        add_input(desc, v.var_node)
        add_input(desc, Tensor(value))
    end
    return Tensor(Operation(desc), 1)
end

"""
Update `ref` by setting its values at `indices` to `updates`.

Args:
* `ref`: The `Variable` to update.
* `indices`: The indices of `ref` to change the values of.
* `updates`: The new values of `ref` at `indices`.
* `use_locking`: Optional `Bool` which, if `true` (default is `false`), protects the assignment with a lock.

Returns:
`ref`, the updated `Variable`.
"""
@op function scatter_update(ref, indices, updates; name=nothing)
    local desc
    with_op_name(name, "ScatterUpdate") do
        desc = NodeDescription("ScatterUpdate")
        add_input(desc, Tensor(ref))
        add_input(desc, Tensor(indices)-1)
        add_input(desc, Tensor(updates))
    end
    Tensor(Operation(desc))
end

for (func, name) in [
    (:scatter_sub, "ScatterSub"),
    (:scatter_add, "ScatterAdd"),
    (:scatter_mul, "ScatterMul"),
    (:scatter_div, "ScatterDiv")]
    @eval begin
        @op function $func(ref, indices, updates; use_locking=false, name=nothing)
            local desc
            with_op_name(name, $name) do
                desc = NodeDescription($name)
                add_input(desc, Tensor(ref))
                add_input(desc, Tensor(indices)-1)
                add_input(desc, Tensor(updates))
                desc["use_locking"] = use_locking
            end
            Tensor(Operation(desc))
        end
    end
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
