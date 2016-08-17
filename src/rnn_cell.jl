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
    B = get_variable("bias", [cell.hidden_size], T)
    X = cat(Node, 2, input, state)
    activity = atan(X*W + B)
    return [activity, activity]
end

type BasicLSTMCell <: RNNCell
    hidden_size::Int
end

output_size(cell::BasicLSTMCell) = div(cell.hidden_size, 2)
state_size(cell::BasicLSTMCell) = cell.hidden_size

function zero_state(cell::BasicLSTMCell, batch_size, T)
    zeros(Node, T, (batch_size, cell.hidden_size))
end

function (cell::BasicLSTMCell)(input, state)
    shape = get_shape(input)
    # TODO finish this
end


end
