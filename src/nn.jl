module nn

import ..TensorFlow: Node, NodeDescription, get_def_graph, capitalize, add_input, Port, get_name, set_attr_list

for f in [:relu, :relu6, :elu, :softplus, :softsign, :softmax, :sigmoid, :tanh]
    @eval function $f(n::Node; name="")
        name = get_name(name)
        desc = NodeDescription(get_def_graph(), $(capitalize(f)), name)
        add_input(desc, Port(n))
        Node(desc)
    end
end

function conv2d(input, filter, strides, padding; data_format="NHWC", name="")
    desc = NodeDescription(get_def_graph(), "Conv2D", get_name(name))
    add_input(desc, Node(input))
    add_input(desc, Node(filter))
    desc["padding"] = padding
    desc["data_format"] = data_format
    set_attr_list(desc, "strides", strides)
    Node(desc)
end

function max_pool(value, ksize, strides, padding; data_format="NHWC", name="")
    desc = NodeDescription(get_def_graph(), "MaxPool", get_name(name))
    add_input(desc, value)
    desc["data_format"] = data_format
    desc["padding"] = padding
    set_attr_list(desc, "ksize", ksize)
    set_attr_list(desc, "strides", strides)
    Node(desc)
end

end
