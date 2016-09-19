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

import TensorFlow
const tf = TensorFlow
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

function dropout(x, keep_prob; noise_shape=nothing, seed=0, name="Dropout")
    local y
    tf.with_op_name(name) do
        keep_prob = Tensor(keep_prob)
        x_scaled = x/keep_prob
        if noise_shape == nothing
            noise_shape = shape(x)
        end
        r = random_uniform(noise_shape, seed=seed, dtype=eltype(x))
        y = x_scaled .* floor(keep_prob+r)
    end
    y
end

function sigmoid_cross_entropy_with_logits(logits, targets; name="")
    #  TODO make numerically stable
    -logits.*targets + log(1+ exp(logits))
end

function sparse_softmax_cross_entropy_with_logits(logits, labels; name="")
    desc = NodeDescription("SparseSoftmaxCrossEntropyWithLogits", get_name(name))
    add_input(desc, Tensor(logits))
    add_input(desc, Tensor(labels)-1)
    Tensor(Operation(desc))
end

function log_softmax(logits; name="")
    desc = NodeDescription("LogSoftmax", get_name(name))
    add_input(desc, logits)
    Tensor(Operation(desc))
end

function embedding_lookup(params, ids; partition_strategy="mod", name="", validate_indices=true)
    ids = Tensor(ids)
    if isa(params, AbstractArray)
        if length(params) > 1
            error("Embedding lookup across multiple parameter tensors not supported yet")
        else
            params = params[1]
        end
    end
    tf.gather(params, ids; name=name)
end

@not_implemented function embedding_lookup_sparse()
end

function top_k(input, k=1; sorted=true, name="")
    desc = NodeDescription("TopKV2", get_name(name))
    add_input(desc, Tensor(input))
    add_input(desc, tf.cast(Tensor(k), Int32))
    desc["sorted"] = sorted
    op = Operation(desc)
    Tensor(op, 1), Tensor(op, 2)+1
end

function in_top_k(predictions, targets, k; name="")
    desc = NodeDescription("InTopK", get_name(name))
    add_input(desc, tf.cast(Tensor(predictions), Float32))
    add_input(desc, Tensor(targets)-1)
    desc["k"] = Int64(k)
    Tensor(Operation(desc))
end

function l2_loss(t, name="")
    reduce_sum(t.*t; name=name)
end

@not_implemented function nce_loss()
end

@not_implemented function sampled_softmax_loss()
end

@not_implemented function batch_normalization(x, mean, variance, offset, scale, variable_epsilon; name="")

end

function local_response_normalization(input; depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name="")
    desc = NodeDescription("LRN", get_name(name))
    desc["depth_radius"] = Int64(depth_radius)
    desc["bias"] = Float32(bias)
    desc["alpha"] = Float32(alpha)
    desc["beta"] = Float32(beta)
    add_input(desc, input)
    Tensor(Operation(desc))
end

@not_implemented function log_uniform_candidate_sampler()
end

@not_implemented function all_candidate_sampler()
end

@not_implemented function atrous_conv2d()
end

@not_implemented function avg_pool()
end

@not_implemented function batch_norm_with_global_normalization()
end

@not_implemented function bias_add()
end

@not_implemented function conv1d()
end

@not_implemented function conv3d()
end

@not_implemented function depthwise_conv2d()
end

@not_implemented function dilation2d()
end

@not_implemented function erosion2d()
end

@not_implemented function fixed_unigram_candidate_sampler()
end

function l2_normalize(x, dim; epsilon=1e-12, name="L2Normalize")
    # TODO take into account epsilon
    local out
    tf.with_op_name(name) do
        sums = tf.reduce_sum(x.*x, reduction_indices=[dim], keep_dims=true)
        norm = sqrt(sums)
        out = x/norm
    end
    out
end

@not_implemented function max_pool3d()
end

@not_implemented function weighted_cross_entropy_with_logits()
end


end
