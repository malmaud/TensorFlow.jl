type Variable <: AbstractNode
    var_node::Node
    assign_node::Node

    Variable() = new()


end

function Variable(initial_value, name="")
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

function assign(v::Variable, value)
    desc = NodeDescription(get_def_graph(), "Assign", get_name())
    add_input(desc, v.var_node)
    add_input(desc, convert(Node, value))
    return Node(desc)
end

Base.convert(::Type{Tensor}, v::Variable) = v.var_node
Base.convert(::Type{Node}, v::Variable) = convert(Tensor, v)

function initialize_all_variables()
    return [var.assign_node for var in get_collection(:Variables)]
end

run(sess::Session, var::Variable) = run(sess, [var])[1]
run(sess::Session, vars::AbstractVector{Variable}) = run(sess, [var.var_node for var in vars])
