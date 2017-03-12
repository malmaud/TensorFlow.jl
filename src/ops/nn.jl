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
batch_normalization,
seq2seq,
conv2d_transpose

using Compat
import TensorFlow
const tf = TensorFlow
import ..TensorFlow: Operation, NodeDescription, get_def_graph, capitalize, add_input, Port, get_name, set_attr_list, get_shape, variable_scope, shape, random_uniform, AbstractTensor, Tensor, reduce_sum, @not_implemented, with_op_name, @op, @tf

for f in [:relu, :relu6, :elu, :softplus, :softsign, :softmax, :sigmoid, :tanh]
    @eval @op function $f(n::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, string($f)) do
            desc = NodeDescription($(capitalize(f)))
            add_input(desc, Tensor(n))
        end
        Tensor(Operation(desc), 1)
    end
end

"""
`conv2d(input, filter, strides, padding; data_format="NHWC")`

Computes a 2-dimensional convolution on 4-dimensional input `Tensor`s `input` and `filter`.

Args:
* `input`: A `Tensor`.
* `filter`: A `Tensor` of the same type as `input`.
* `strides`: A list of `Int`s controlling the stride of the sliding window.
* `padding`: A string, either `'VALID'` or `'SAME'`. Specifies which padding algorithm to use.
* `data_format`: A string specifying which data format to use. The default is `'NHWC'`. The other option is `'NCHW'`.
"""
@op function conv2d(input, filter, strides, padding; data_format="NHWC", name=nothing)
    local desc
    with_op_name(name, "Conv2D") do
        desc = NodeDescription("Conv2D")
        add_input(desc, Tensor(input))
        add_input(desc, Tensor(filter))
        desc["padding"] = padding
        desc["data_format"] = data_format
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

"""
`max_pool(value, ksize, strides, padding; data_format="NHWC")`

Performs max pooling on the input.

Args:
* `value`: A 4-dimensional `Tensor` to pool.
* `ksize`: A list of `Int`s at least length 4. The size of the pooling window in each dimension.
* `strides`: A list of `Int`s at least length 4. The stride of the sliding pooling window.
* `padding`: A string, either `'VALID'` or `'SAME'`. Specifies which padding algorithm to use.
* `data_format`: A string specifying which data format to use. The default is `'NHWC'`. The other option is `'NCHW'`.
"""
@op function max_pool(value, ksize, strides, padding; data_format="NHWC", name=nothing)
    local desc
    with_op_name(name, "MaxPool") do
        desc = NodeDescription("MaxPool")
        add_input(desc, value)
        desc["data_format"] = data_format
        desc["padding"] = padding
        set_attr_list(desc, "ksize", ksize)
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

include("rnn_cell.jl")
import .rnn_cell:  zero_state, output_size, state_size

"""
`rnn(cell, inputs; initial_state=nothing, dtype=nothing, sequence_length=nothing, scope="RNN")`

Creates a recurrent neural network.

Args:
* `cell`: An instance of `RNNCell`.
* `inputs`: A list of input `Tensor`s, each with size `(batch_size, input_size)`.
* `initial_state`: A starting state for the RNN. If not provided, the initial state is `zero_state`.
* `dtype`: Data type of the initial state and output, in case it cannot be inferred.
* `sequence_length`: Specifies length of each sequence in `inputs`.
* `scope`: `VariableScope` for the subgraph. Defaults to `RNN`.
"""
function rnn(cell, inputs; initial_state=nothing, dtype=nothing, sequence_length=nothing, scope="RNN")
    # TODO use sequence length
    if initial_state === nothing
        if dtype === nothing
            error("dtype must be set if initial_state is not provided")
        end
        batch_size = get_shape(first(inputs), 1)
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

"""
`dynamic_rnn(cell, inputs; sequence_length=nothing, initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN")`

Creates a *dynamic* recurrent neural network. Performs full dynamic unrolling of `inputs`.

Args:
* `cell`: An instance of `RNNCell`.
* `inputs`: A `Tensor` of shape `[max_time, batch_size, ..., ...]` or `[batch_size, max_time, ..., ...]` (see `time_major`). May also be a nested `Tuple` with the same property. The first two dimensions *must* be the same across
all elements but all later dimensions may vary.
* `sequence_length`: Specifies length of each sequence in `inputs`.
* `initial_state`: A starting state for the RNN. If not provided, the initial state is `zero_state`.
* `dtype`: Data type of the initial state and output, in case it cannot be inferred.
* `parallel_iterations`: Number of iterations to run in parallel, defaulting to `32`. Ops that have no temporal dependency can be run in parallel and will be. Trades time for space - larger values use more memory.
* `swap_memory`: Optional `Bool`, which if `true` allows transparent swapping of `Tensor`s needed for back propagation from the CPU to the GPU and vice versa. Allows training of RNNs which wouldn't fit on a single GPU. Defaults to `false`.
* `time_major`: Shape format for `inputs` and `outputs` `Tensor`s. Determines whether the first dimension of each is `max_time` (`true`) or `batch_size` (`false`, default). `true` is more efficient but is the transpose of most TensorFlow operations.
* `scope`: `VariableScope` for the subgraph. Defaults to `RNN`.
"""
@op function dynamic_rnn(cell, inputs; sequence_length=nothing, initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN")
    sequence_length === nothing || error("sequence_length parameter not supported yet")
    time_major == false || error("Time-major order not supported yet")
    time_step = tf.constant(1)
    num_steps = tf.cast(tf.shape(inputs)[2], Int64)

    if initial_state === nothing
        if dtype === nothing
            error("dtype must be set if initial_state is not provided")
        end
        batch_size = get_shape(first(inputs), 1)
        initial_state = zero_state(cell, batch_size, dtype)
    end

    state = initial_state
    input_dim = get_shape(inputs, 3)
    
    output = tf.zeros(Tensor, eltype(state), (get_shape(inputs, 1), output_size(cell)))

    while_output = @tf while time_step ≤ num_steps
        slice_start = tf.pack([0, time_step-1, 0])
        slice_size = tf.pack([-1, 1, -1])
        data = tf.slice(inputs, slice_start, slice_size)
        data = tf.squeeze(data, [2])
        local new_state
        variable_scope(scope) do
            output, new_state = cell(data, state, input_dim)
        end
        [time_step=>time_step+1, state=>new_state, output=>output]
    end

    final_state = while_output[2]
    final_output = while_output[3]
    final_output, final_state
end

@not_implemented function state_saving_rnn()
end

@not_implemented function bidirectional_rnn()
end

"""
`dropout(x, keep_prob; noise_shape=nothing, seed=0)`

Keeps each element of `x` with `keep_prob`, scaled by `1/keep_prob`, otherwise outputs `0`.
This computes dropout.

Args:
* `x`: A `Tensor`.
* `keep_prob`: Probability that each element is kept.
* `noise_shape`: Shape for randomly generated keep/drop flags.
* `seed`: Integer used to seed the RNG. Defaults to `0`.
"""
@op function dropout(x, keep_prob; noise_shape=nothing, seed=0, name=nothing)
    local y
    with_op_name(name, "Dropout") do
        keep_prob = tf.cast(Tensor(keep_prob), eltype(x))
        x_scaled = x/keep_prob
        if noise_shape == nothing
            noise_shape = shape(x)
        end
        r = random_uniform(noise_shape, 0, 1, seed=seed, dtype=eltype(x))
        y = x_scaled .* floor(keep_prob+r)
    end
    y
end

@op function sigmoid_cross_entropy_with_logits(logits, targets; name=nothing)
    #  TODO make numerically stable
    local out
    with_op_name(name, "SigmoidCrossEntropyWithLogits") do
        out = -logits.*targets + log(1+ exp(logits))
    end
    out
end

@op function sparse_softmax_cross_entropy_with_logits(logits, labels; name=nothing)
    local desc
    with_op_name(name, "SparseSoftmaxCrossEntropyWithLogits") do
        desc = NodeDescription("SparseSoftmaxCrossEntropyWithLogits")
        add_input(desc, Tensor(logits))
        add_input(desc, Tensor(labels)-1)
    end
    Tensor(Operation(desc))
end

@op function log_softmax(logits; name=nothing)
    local desc
    with_op_name(name, "LogSoftmax") do
        desc = NodeDescription("LogSoftmax")
        add_input(desc, logits)
    end
    Tensor(Operation(desc))
end

"""
    embedding_lookup(params, ids; partition_strategy="mod", name="", validate_indices=true)

Looks up values of `ids` in `params`. Currently only supports one `Tensor` in `params`.

Args:
* `params`: A list of `Tensor`s of the same type and which can be concatenated along their first dimension.
* `ids`: A `Tensor` of `Int32` or `Int64` ids to be looked up in `params`.
* `partition_strategy`: If `"mod"` (default), assign each id to partition `p = id % len(params)`. If `"div"`, assign each id contiguously.
* `name`: An optional name for the operation.
* `validate_indices`: If `true` (default), make sure the indices are valid.
"""
@op function embedding_lookup(params, ids; partition_strategy="mod", name=nothing, validate_indices=true)
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

"""
`top_k(input, k=1; sorted=true)`

Finds values and indices of the top `k` largest entries of `input` in its
last dimension.

Args:
* `input`: One or more dimensional `Tensor` with last dimension at least size `k`.
* `k`: Number of largest elements of `input` to look for. Defaults to 1.
* `sorted`: If `true` (default), the returned values will be sorted in descending order.
"""
@op function top_k(input, k=1; sorted=true, name=nothing)
    local desc
    with_op_name(name, "TopKV2") do
        desc = NodeDescription("TopKV2")
        add_input(desc, Tensor(input))
        add_input(desc, tf.cast(Tensor(k), Int32))
        desc["sorted"] = sorted
    end
    op = Operation(desc)
    Tensor(op, 1), Tensor(op, 2)+1
end

"""
`in_top_k(predictions, targets, k)`

Determines whether the `targets` are in the top `k` `predictions`.
Outputs a `batch_size` (first dimension of `predictions`) array of boolean values.

Args:
* `predictions`: Two dimensional `Tensor`.
* `targets`: A `Tensor`.
* `k`: Number of elements to look at for comparison.
"""
@op function in_top_k(predictions, targets, k; name=nothing)
    local desc
    with_op_name(name, "InTopK") do
        desc = NodeDescription("InTopK")
        add_input(desc, cast(Tensor(predictions), Float32))
        add_input(desc, Tensor(targets)-1)
        desc["k"] = Int64(k)
    end
    Tensor(Operation(desc))
end

"""
`l2_loss(t)`

Computes half the L2-norm of a `Tensor` `t`, without taking the square root.
"""
@op function l2_loss(t; name=nothing)
    local out
    with_op_name(name, "L2_Loss") do
        out = sqrt(reduce_sum(t.*t))
    end
    out
end

@not_implemented function nce_loss()
end

@not_implemented function sampled_softmax_loss()
end

@not_implemented function log_poisson_loss()
end

@not_implemented function batch_normalization(x, mean, variance, offset, scale, variable_epsilon; name="")

end

@op function local_response_normalization(input; depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=nothing)
    local desc
    with_op_name(name, "LRN") do
        desc = NodeDescription("LRN")
        desc["depth_radius"] = Int64(depth_radius)
        desc["bias"] = Float32(bias)
        desc["alpha"] = Float32(alpha)
        desc["beta"] = Float32(beta)
        add_input(desc, input)
    end
    Tensor(Operation(desc))
end

@not_implemented function log_uniform_candidate_sampler()
end

@not_implemented function all_candidate_sampler()
end

@op function atrous_conv2d(value, filters, rate, padding; name=nothing)
    local desc
    with_op_name(name, "AtrousConv2D") do
        desc = NodeDescription("AtrousConv2D")
        add_input(desc, Tensor(value))
        add_input(desc, Tensor(filter))
        desc["padding"] = padding
        desc["rate"]    = rate
    end
    Tensor(Operation(desc), 1)
end

@op function avg_pool(value, ksize, strides, padding; data_format="NHWC", name=nothing)
    local desc
    with_op_name(name, "AvgPool") do
        desc = NodeDescription("AvgPool")
        add_input(desc, value)
        desc["data_format"] = data_format
        desc["padding"] = padding
        set_attr_list(desc, "ksize", ksize)
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@not_implemented function batch_norm_with_global_normalization()
end

@not_implemented function bias_add()
end

@op function conv1d(value, filters, strides, padding; data_format="NHWC", name=nothing)
    local desc
    with_op_name(name, "Conv1D") do
        desc = NodeDescription("Conv1D")
        add_input(desc, Tensor(value))
        add_input(desc, Tensor(filters))
        desc["padding"] = padding
        desc["data_format"] = data_format
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@op function conv3d(input, filter, strides, padding; name=nothing)
    local desc
    with_op_name(name, "Conv3D") do
        desc = NodeDescription("Conv3D")
        add_input(desc, Tensor(input))
        add_input(desc, Tensor(filter))
        desc["padding"] = padding
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@op function depthwise_conv2d(input, filter, strides, padding; name=nothing)
    local desc
    with_op_name(name, "DepthwiseConv2D") do
        desc = NodeDescription("DepthwiseConv2D")
        add_input(desc, Tensor(input))
        add_input(desc, Tensor(filter))
        desc["padding"] = padding
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@op function dilation2d(input, filter, strides, rates, padding; name=nothing)
    local desc
    with_op_name(name, "Dilation2D") do
        desc = NodeDescription("Dilation2D")
        add_input(desc, Tensor(input))
        add_input(desc, Tensor(filter))
        desc["padding"] = padding
        set_attr_list(desc, "rates", rates)
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@op function erosion2d(value, kernel, strides, rates, padding; name=nothing)
    local desc
    with_op_name(name, "Erosion2D") do
        desc = NodeDescription("Erosion2D")
        add_input(desc, Tensor(value))
        add_input(desc, Tensor(kernel))
        desc["padding"] = padding
        set_attr_list(desc, "rates", rates)
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@not_implemented function fixed_unigram_candidate_sampler()
end

@op function l2_normalize(x, dim; epsilon=1e-12, name=nothing)
    # TODO take into account epsilon
    local out
    tf.with_op_name(name, "L2Normalize") do
        sums = tf.reduce_sum(x.*x, reduction_indices=[dim], keep_dims=true)
        norm = sqrt(sums)
        out = x/norm
    end
    out
end

@op function max_pool3d(input, ksize, strides, padding; name=nothing)
    local desc
    with_op_name(name, "MaxPool3D") do
        desc = NodeDescription("MaxPool3D")
        add_input(desc, input)
        desc["padding"] = padding
        set_attr_list(desc, "ksize", ksize)
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@op function avg_pool3d(input, ksize, strides, padding; name=nothing)
    local desc
    with_op_name(name, "AvgPool3D") do
        desc = NodeDescription("AvgPool3D")
        add_input(desc, input)
        desc["padding"] = padding
        set_attr_list(desc, "ksize", ksize)
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc), 1)
end

@op function weighted_cross_entropy_with_logits(logits, targets, pos_weight; name=nothing)
    local desc
    with_op_name(name, "WeightedCrossEntropyWithLogits") do
        desc = NodeDescription("WeightedCrossEntropyWithLogits")
        add_input(desc, Tensor(logits))
        add_input(desc, Tensor(targets))
        add_input(desc, Tensor(pos_weight))
    end
    Tensor(Operation(desc))
end

@op function conv2d_transpose(value, filter, output_shape, strides; padding="SAME", data_format="NHWC", name=nothing)
    local desc
    with_op_name(name, "conv2d_transpose") do
        desc = NodeDescription("Conv2DBackpropInput")
        add_input(desc, Tensor(output_shape))
        add_input(desc, Tensor(filter))
        add_input(desc, Tensor(value))
        desc["data_format"] = data_format
        desc["padding"] = padding
        set_attr_list(desc, "strides", strides)
    end
    Tensor(Operation(desc))
end

include("seq2seq.jl")

end
