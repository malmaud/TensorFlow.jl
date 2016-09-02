module nn

export
conv2d,
max_pool,
zero_state,
output_size,
state_size,
rnn,
dynamic_rnn,
dropout,
relu,
relu6,
elu,
softplus,
softsign,
softmax,
sigmoid,
tanh,
state_saving_rnn,
bidirectional_rnn,
sigmoid_cross_entropy_with_logits,
sparse_softmax_cross_entropy_with_logits,
log_softmax,
embedding_lookup,
top_k,
in_top_k,
l2_loss,
log_poisson_loss,
nce_loss,
sampled_softmax_loss,
batch_normalization


import ..TensorFlow: Operation, NodeDescription, get_def_graph, capitalize, add_input, Port, get_name, set_attr_list, get_shape, variable_scope, shape, random_uniform, AbstractTensor, Tensor, reduce_sum, @not_implemented

for f in [:relu, :relu6, :elu, :softplus, :softsign, :softmax, :sigmoid, :tanh]
    @eval function $f(n::AbstractTensor; name="")
        name = get_name(name)
        desc = NodeDescription($(capitalize(f)), name)
        add_input(desc, Tensor(n))
        Tensor(Operation(desc), 1)
    end
end

function conv2d(input, filter, strides, padding; data_format="NHWC", name="")
    desc = NodeDescription("Conv2D", get_name(name))
    add_input(desc, Tensor(input))
    add_input(desc, Tensor(filter))
    desc["padding"] = padding
    desc["data_format"] = data_format
    set_attr_list(desc, "strides", strides)
    Tensor(Operation(desc), 1)
end

function max_pool(value, ksize, strides, padding; data_format="NHWC", name="")
    desc = NodeDescription("MaxPool", get_name(name))
    add_input(desc, value)
    desc["data_format"] = data_format
    desc["padding"] = padding
    set_attr_list(desc, "ksize", ksize)
    set_attr_list(desc, "strides", strides)
    Tensor(Operation(desc), 1)
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
    outputs = Tensor[]
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

@not_implemented function dynamic_rnn(cell, inputs; sequence_length=nothing, initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN")

end

@not_implemented function state_saving_rnn()
end

@not_implemented function bidirectional_rnn()
end

function dropout(x, keep_prob; noise_shape=nothing, seed=0, name="")
    keep_prob = Tensor(keep_prob)
    x_scaled = x/keep_prob
    if noise_shape == nothing
        noise_shape = shape(x)
    end
    r = random_uniform(noise_shape, seed=seed, dtype=eltype(x))
    y = x_scaled .* floor(keep_prob+r)
end

@not_implemented function sigmoid_cross_entropy_with_logits()
end

@not_implemented function sparse_softmax_cross_entropy_with_logits()

end

@not_implemented function log_softmax()
end

function embedding_lookup(params, ids; partition_strategy="mod", name="", validate_indices=true)
    if ndims(params) > 0
        error("Embedding lookup across multiple parameter tensors not supported yet")
    end
    gather(params, id; name=name)
end

@not_implemented function embedding_lookup_sparse()
end

@not_implemented function top_k()
end

@not_implemented function in_top_k()
end

function l2_loss(t, name="")
    reduce_sum(t.*t; name=name)
end

@not_implemented function log_poisson_loss()

end

@not_implemented function nce_loss()
end

@not_implemented function sampled_softmax_loss()
end

@not_implemented function batch_normalization()
end

end
