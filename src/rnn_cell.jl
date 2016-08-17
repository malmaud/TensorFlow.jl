module rnn_cell

import ...TensorFlow: Node, get_shape, get_variable

abstract RNNCell

type BasicRNNCell <: RNNCell
    hidden_size::Int
end

function zero_state(cell::BasicRNNCell, batch_size, T)
    zeros(Node, T, (batch_size, cell.hidden_size))
end

output_size(cell::BasicRNNCell) = cell.hidden_size
state_size(cell::BasicRNNCell) = cell.hidden_size

function (cell::BasicRNNCell)(input, state)
    shape = get_shape(input)
    N = shape[2] + cell.hidden_size
    batch_size = shape[1]
    T = eltype(state)
    W = get_variable("weights", [N, cell.hidden_size], T)
    X = cat(Node, 2, input, state)
    activity = X*W
    return activity
end


end
