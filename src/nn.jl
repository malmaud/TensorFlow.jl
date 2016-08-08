module nn

import ..TensorFlow: Node, NodeDescription, get_def_graph, capitalize, add_input, Port, get_name

for f in [:relu, :relu6, :elu, :softplus, :softsign, :softmax]
    @eval function $f(n::Node, name="")
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $(capitalize(f)), name)
        add_input(desc, Port(n))
        Node(desc)
    end
end


end
