module rnn_cell

export
zero_state,
output_size,
state_size,
LSTMCell,
GRUCell,
BasicRNNCell,
RNNCell

import ...TensorFlow: Operation, get_shape, get_variable, tanh, Tensor
import ..nn: sigmoid

abstract RNNCell

type BasicRNNCell <: RNNCell
    hidden_size::Int
end

function check_shape(shape)
    if shape.rank_unknown || isnull(shape.dims[2])
        error("Shape must be inferable")
    end
end

function zero_state(cell::RNNCell, batch_size, T)
    zeros(Tensor, T, (batch_size, state_size(cell)))
end

output_size(cell::BasicRNNCell) = cell.hidden_size
state_size(cell::BasicRNNCell) = cell.hidden_size

function (cell::BasicRNNCell)(input, state)
    shape = get_shape(input)
    check_shape(shape)
    N = get(shape.dims[2]) + cell.hidden_size
    T = eltype(state)
    W = get_variable("weights", [N, cell.hidden_size], T)
    B = get_variable("bias", [cell.hidden_size], T)
    X = cat(Tensor, 2, input, state)
    activity = tanh(X*W + B)
    return [activity, activity]
end

type LSTMCell <: RNNCell
    hidden_size::Int
end

output_size(cell::LSTMCell) = cell.hidden_size

immutable LSTMStateTuple
    c
    h
end

function Base.show(io::IO, s::LSTMStateTuple)
    print(io, "LSTMStateTuple(c=$(s.c), h=$(s.h))")
end

state_size(cell::LSTMCell) = LSTMStateTuple(cell.hidden_size, cell.hidden_size)

function zero_state(cell::LSTMCell, batch_size, T)
    LSTMStateTuple(zeros(Tensor, T, (batch_size, cell.hidden_size)),
        zeros(Tensor, T, (batch_size, cell.hidden_size)))
end

function (cell::LSTMCell)(input, state)
    shape = get_shape(input)
    check_shape(shape)
    N = get(shape.dims[2]) + cell.hidden_size
    T = eltype(state)
    X = cat(Tensor, 2, input, state.h)

    Wi = get_variable("Wi", [N, cell.hidden_size], T)
    Wf = get_variable("Wf", [N, cell.hidden_size], T)
    Wo = get_variable("Wo", [N, cell.hidden_size], T)
    Wg = get_variable("Wg", [N, cell.hidden_size], T)

    Bi = get_variable("Bi", [cell.hidden_size], T)
    Bf = get_variable("Bf", [cell.hidden_size], T)
    Bo = get_variable("Bo", [cell.hidden_size], T)
    Bg = get_variable("Bg", [cell.hidden_size], T)

    # TODO make this all one multiply
    I = sigmoid(X*Wi + Bi)
    F = sigmoid(X*Wf + Bf)
    O = sigmoid(X*Wo + Bo)
    G = tanh(X*Wg + Bg)
    C = state.c.*F + G.*I
    S = tanh(C).*O

    return (S, LSTMStateTuple(C, S))
end

type GRUCell <: RNNCell
    hidden_size::Int
end

output_size(cell::GRUCell) = cell.hidden_size
state_size(cell::GRUCell) = cell.hidden_size

function (cell::GRUCell)(input, state)
    T = eltype(state)
    shape = get_shape(input)
    check_shape(shape)
    N = get(shape.dims[2]) + cell.hidden_size
    X = cat(Tensor, 2, input, state)
    Wz = get_variable("Wz", [N, cell.hidden_size], T)
    Wr = get_variable("Wr", [N, cell.hidden_size], T)
    Wh = get_variable("Wh", [N, cell.hidden_size], T)
    Bz = get_variable("Bz", [cell.hidden_size], T)
    Br = get_variable("Br", [cell.hidden_size], T)
    Bh = get_variable("Bh", [cell.hidden_size], T)
    z = sigmoid(X*Wz + Bz)
    r = sigmoid(X*Wr + Br)
    X2 = cat(Tensor, 2, input, state.*r)
    h = sigmoid(X2*Wh + Bh)
    s2 = (1-z).*h + z.*state
    return [s2, s2]
end

type MultiRNNCell <: RNNCell
    cells::Vector
    state_is_tuple::Bool
end

function output_size(cell::MultiRNNCell)
    output_size(cell.cells[end])
end

function state_size(cell::MultiRNNCell)
    # TODO
end

function zero_state(cell::MultiRNNCell, batch_size, T)
    # TODO
end

function (cell::MultiRNNCell)(input, state)
    for subcell in cell.cells
        input, state = subcell(input, state)
    end
    input, state
end

end
