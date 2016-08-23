module rnn_cell

import ...TensorFlow: Operation, get_shape, get_variable, tanh
import ..nn: sigmoid

abstract RNNCell

type BasicRNNCell <: RNNCell
    hidden_size::Int
end

function zero_state(cell::BasicRNNCell, batch_size, T)
    zeros(Operation, T, (batch_size, cell.hidden_size))
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
    X = cat(Operation, 2, input, state)
    activity = tanh(X*W + B)
    return [activity, activity]
end

type BasicLSTMCell <: RNNCell
    hidden_size::Int
end

output_size(cell::BasicLSTMCell) = div(cell.hidden_size, 2)
state_size(cell::BasicLSTMCell) = cell.hidden_size

function zero_state(cell::BasicLSTMCell, batch_size, T)
    zeros(Operation, T, (batch_size, cell.hidden_size))
end

function (cell::BasicLSTMCell)(input, state)
    shape = get_shape(input)
    # TODO finish this
end

type GRUCell <: RNNCell
    hidden_size::Int
end

output_size(cell::GRUCell) = cell.hidden_size
state_size(cell::GRUCell) = cell.hidden_size

function zero_state(cell::GRUCell, batch_size, T)
    zeros(Operation, T, (batch_size, cell.hidden_size))
end

function (cell::GRUCell)(input, state)
    T = eltype(state)
    shape = get_shape(input)
    N = shape[2] + cell.hidden_size
    X = cat(Operation, 2, input, state)
    Wz = get_variable("Wz", [N, cell.hidden_size], T)
    Wr = get_variable("Wr", [N, cell.hidden_size], T)
    Wh = get_variable("Wh", [N, cell.hidden_size], T)
    Bz = get_variable("Bz", [cell.hidden_size], T)
    Br = get_variable("Br", [cell.hidden_size], T)
    Bh = get_variable("Bh", [cell.hidden_size], T)
    z = sigmoid(X*Wz + Bz)
    r = sigmoid(X*Wr + Br)
    X2 = cat(Operation, 2, input, state.*r)
    h = sigmoid(X2*Wh + Bh)
    s2 = (1-z).*h + z.*state
    return [s2, s2]
end


end
