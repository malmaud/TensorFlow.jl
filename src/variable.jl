using Distributions

type Variable <: AbstractNode
    var_node::Node
    assign_node::Node

    Variable() = new()
end

function Variable(initial_value; name="")
    self = Variable()

    name = get_name(name)
    desc = NodeDescription(get_def_graph(), "Variable", name)
    desc["dtype"] = eltype(initial_value)
    desc["shape"] = size(initial_value)
    self.var_node = Node(desc)

    desc = NodeDescription(get_def_graph(), "Assign", "$name/Assign")
    add_input(desc, self.var_node)
    add_input(desc, convert(Node, initial_value))
    self.assign_node = Node(desc)

    add_to_collection(:Variables, self)
    return self
end

function get_shape(v::Variable)
    return get_shape(Operation(v.assign_node).inputs[2])
end

function assign(v::Variable, value)
    desc = NodeDescription(get_def_graph(), "Assign", get_name())
    add_input(desc, v.var_node)
    add_input(desc, convert(Node, value))
    return Node(desc)
end

Base.setindex!(v::Variable, value) = assign(v, value)

Base.convert(::Type{Tensor}, v::Variable) = v.var_node
Base.convert(::Type{Node}, v::Variable) = convert(Tensor, v)

function initialize_all_variables()
    return [var.assign_node for var in get_collection(:Variables)]
end

run(sess::Session, var::Variable) = run(sess, [var])[1]
run(sess::Session, vars::AbstractVector{Variable}) = run(sess, [var.var_node for var in vars])

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

function variable_scope(f, name; kwargs...)
    scope = make_scope(name; kwargs...)
    push!(scope_stack, scope)
    try
        f()
    finally
        pop!(scope_stack)
    end
end

function get_variable(var_name, shape, dtype; kwargs...)
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
            v = Variable(map(dtype, rand(initializer, shape...)), name)
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
