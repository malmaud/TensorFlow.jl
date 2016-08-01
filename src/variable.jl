type Variable
    var_node::Node
    assign_node::Node

    function Variable(initial_value, name="")
        self = new()

        name = get_name(name)
        desc = NodeDescription(get_def_graph(), "Variable", name)
        desc["dtype"] = eltype(initial_value)
        desc["shape"] = size(initial_value)
        self.var_node = Node(desc)

        desc = NodeDescription(get_def_graph(), "Assign", "$name/Assign")
        add_input(desc, self.var_node)
        add_input(desc, constant(initial_value))  # TODO: let this be a node
        self.assign_node = Node(desc)

        add_to_collection(:Variables, self)
        return self
    end
end

function initialize_all_variables()
    return [var.assign_node for var in get_collection(:Variables)]
end

run(sess::Session, var::Variable) = run(sess, [var])[1]
run(sess::Session, vars::AbstractVector{Variable}) = run(sess, [var.var_node for var in vars])
