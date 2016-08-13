module rnn_cell

import ...TensorFlow: Node

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
    cell.hidden_size
end


end
