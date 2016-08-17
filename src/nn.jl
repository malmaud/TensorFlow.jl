module nn

import ..TensorFlow: Node, NodeDescription, get_def_graph, capitalize, add_input, Port, get_name, set_attr_list, get_shape, variable_scope

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

include("rnn_cell.jl")
import .rnn_cell:  zero_state, output_size, state_size

function rnn(cell, inputs; initial_state=nothing, dtype=nothing, sequence_length=nothing, scope="RNN")
    # TODO use sequence length
    if initial_state === nothing
        if dtype === nothing
            error("dtype must be set if initial_state is not provided")
        end
        shape = get_shape(inputs[1])
        if shape == -1
            error("Shape of input is unknown")
        end
        batch_size = shape[1]
        initial_state = zero_state(cell, batch_size, dtype)
    end
    outputs = Node[]
    local output
    state = initial_state
    for (idx, input) in enumerate(inputs)
        variable_scope(scope; reuse=idx>1) do
            output, state = cell(input, state)
        end
        push!(outputs, output)
    end
    return outputs, state
end

function dynamic_rnn(cell, inputs; sequence_length=nothing, initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN")
    error("Not implemented yet")
end

end
