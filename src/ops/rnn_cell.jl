module rnn_cell

export
zero_state,
output_size,
state_size,
LSTMCell,
GRUCell,
BasicRNNCell,
RNNCell

using Compat
import ....Main: TensorFlow
import .TensorFlow: Operation, get_shape, get_variable, tanh, Tensor, nn
import .nn: sigmoid
const tf = TensorFlow

@compat abstract type RNNCell end

type BasicRNNCell <: RNNCell
    hidden_size::Int
end


"""
Form a `RNNCell` with all states initialized to zero.
"""
function zero_state(cell::RNNCell, batch_size, T)
    zeros(Tensor, T, (batch_size, state_size(cell)))
end

output_size(cell::BasicRNNCell) = cell.hidden_size
state_size(cell::BasicRNNCell) = cell.hidden_size

function (cell::BasicRNNCell)(input, state)
    N = get_shape(input, 2) + cell.hidden_size
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

Base.eltype(l::LSTMStateTuple) = eltype(l.c)

function tf.get_tensors(s::LSTMStateTuple)
    [s.c, s.h]
end

function tf.build_output(s::LSTMStateTuple, values, pos=Ref(1))
    out = LSTMStateTuple(values[pos[]], values[pos[]+1])
    pos[] += 2
    out
end

function tf.get_inputs(value::LSTMStateTuple, input_tensors, input_set=[])
    push!(input_set, value.c)
    push!(input_set, value.h)
end

function Base.show(io::IO, s::LSTMStateTuple)
    print(io, "LSTMStateTuple(c=$(s.c), h=$(s.h))")
end

state_size(cell::LSTMCell) = LSTMStateTuple(cell.hidden_size, cell.hidden_size)

"""
Form a `LSTMCell` with all states initialized to zero.
"""
function zero_state(cell::LSTMCell, batch_size, T)
    LSTMStateTuple(zeros(Tensor, T, (batch_size, cell.hidden_size)),
        zeros(Tensor, T, (batch_size, cell.hidden_size)))
end

function (cell::LSTMCell)(input, state)
    N = get_shape(input, 2) + cell.hidden_size
    T = eltype(state)
    X = cat(Tensor, 2, input, state.h)

    Wi = get_variable("Wi", [N, cell.hidden_size], T)
    Wf = get_variable("Wf", [N, cell.hidden_size], T)
    Wo = get_variable("Wo", [N, cell.hidden_size], T)
    Wg = get_variable("Wg", [N, cell.hidden_size], T)

    local Bi, Bf, Bo, Bg
    tf.variable_scope("Bias", initializer=tf.ConstantInitializer(0.0)) do
        Bi = get_variable("Bi", [cell.hidden_size], T)
        Bf = get_variable("Bf", [cell.hidden_size], T)
        Bo = get_variable("Bo", [cell.hidden_size], T)
        Bg = get_variable("Bg", [cell.hidden_size], T)
    end

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
    N = get_shape(input, 2) + cell.hidden_size
    X = cat(Tensor, 2, input, state)
    Wz = get_variable("Wz", [N, cell.hidden_size], T)
    Wr = get_variable("Wr", [N, cell.hidden_size], T)
    Wh = get_variable("Wh", [N, cell.hidden_size], T)
    local Bz, Br, Bh
    tf.variable_scope("Bias", initializer=tf.ConstantInitializer(0.0)) do  # TODO doublecheck python also uses 0 for GRU
        Bz = get_variable("Bz", [cell.hidden_size], T)
        Br = get_variable("Br", [cell.hidden_size], T)
        Bh = get_variable("Bh", [cell.hidden_size], T)
    end
    z = sigmoid(X*Wz + Bz)
    r = sigmoid(X*Wr + Br)
    X2 = cat(Tensor, 2, input, state.*r)
    h = nn.tanh(sigmoid(X2*Wh + Bh))
    s2 = (1-z).*h + z.*state
    return [s2, s2]
end

type MultiRNNCell <: RNNCell
    cells::Vector{RNNCell}
end

function output_size(cell::MultiRNNCell)
    output_size(cell.cells[end])
end

function state_size(cell::MultiRNNCell)
    map(state_size, cell.cells)
end

function zero_state(cell::MultiRNNCell, batch_size, T)
    [zero_state(subcell, batch_size, T) for subcell in cell.cells]
end

function (cell::MultiRNNCell)(input, state)
    states = []
    for (i, (subcell, substate)) in enumerate(zip(cell.cells, state))
        tf.variable_scope("cell$i") do
            input, state = subcell(input, substate)
        end
        push!(states, state)
    end
    input, states
end

type DropoutWrapper{CellType<:RNNCell} <: RNNCell
    cell::CellType
    output_keep_prob::Tensor
end

DropoutWrapper(cell; output_keep_prob=1.0) = DropoutWrapper(cell, Tensor(output_keep_prob))

output_size(cell::DropoutWrapper) = output_size(cell.cell)
state_size(cell::DropoutWrapper) = state_size(cell.cell)
zero_state(cell::DropoutWrapper, batch_size, T) = zero_state(cell.cell, batch_size, T)

function (wrapper::DropoutWrapper)(input, state)
    output, new_state = wrapper.cell(input, state)
    dropped_output = nn.dropout(output, wrapper.output_keep_prob)
    dropped_output, new_state
end


end
