module nn

using Compat
import TensorFlow
const tf = TensorFlow
import .tf: Ops, @op, @not_implemented, @tf

import .Ops:
    relu,
    relu6,
    elu,
    softplus,
    softsign,
    softmax,
    sigmoid,
    conv3d,
    max_pool,
    max_pool3d,
    avg_pool,
    avg_pool3d,
    log_softmax,
    dilation2d,
    conv2d

include("rnn_cell.jl")
import .rnn_cell:  zero_state, output_size, state_size

# Ops.conv2d takes padding and strides as a keyword instead of a positional argument
function conv2d(input, filter, strides, padding; kwargs...)
    conv2d(input, filter; padding=padding, strides=strides, kwargs...)
end

# Same for max pool
function max_pool(input, ksize, strides, padding; kwargs...)
    max_pool(input; ksize=ksize, strides=strides, padding=padding, kwargs...)
end

function conv2d_transpose(value, filter, output_shape, strides; padding="SAME", data_format="NHWC", kwargs...)
    Ops.conv2d_backprop_input(output_shape, filter, value; strides=strides, padding=padding, data_format=data_format, kwargs...)
end

"""
`rnn(cell, inputs; initial_state=nothing, dtype=nothing, sequence_length=nothing, scope="RNN")`

Creates a recurrent neural network.

Args:
* `cell`: An instance of `RNNCell`.
* `inputs`:
    * A Vector of input `Tensor`s, each with size `(batch_size, input_size)`, with the length `max_steps`
    * or, a Tensor of inputs with,  of shape `[max_time, batch_size, ..., ...]` as per `dynamic_rnn` (NB: this feature is not present in the Python TensorFlow client)
* `initial_state`: A starting state for the RNN. If not provided, the initial state is `zero_state`.
* `dtype`: Data type of the initial state and output, in case it cannot be inferred.
* `sequence_length`: Specifies length of each sequence in `inputs`.
* `scope`: `VariableScope` for the subgraph. Defaults to `RNN`.
"""
function rnn(cell, inputs::Vector, sequence_length=nothing; initial_state=nothing, dtype=nothing, scope="RNN")
    if initial_state === nothing
        initial_state = zero_state(cell, first(inputs), dtype)
    end
    outputs = tf.Tensor[]
    state = initial_state
    for (time_step, input) in enumerate(inputs)
        tf.variable_scope(scope; reuse=time_step>1) do
            new_output, new_state = cell(input, state)
        end
        if sequence_length!==nothing && time_step > 1 # This should be removed by the julia lowering process
            # Only update output and state for rows that are not yet passed their ends
            have_passed_end = sequence_length .< time_step
            new_output = tf.select(have_passed_end, outputs[end], new_output)
            new_state = tf.select(have_passed_end, state, new_state)
        end
        state = new_state
        push!(outputs, new_output)
    end
    return outputs, state
end

function rnn(cell, inputs::tf.Tensor, sequence_length=nothing; time_major=false, kwargs...)
    input_list = tf.unstack(inputs; axis = (time_major ? 1 : 2))
    rnn(cell, input_list, sequence_length; kwargs...)
end

"""
`dynamic_rnn(cell, inputs, sequence_length, initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN")`

Creates a *dynamic* recurrent neural network. Performs full dynamic unrolling of `inputs`.

Args:
* `cell`: An instance of `RNNCell`.
* `inputs`: A `Tensor` of shape `[max_time, batch_size, ..., ...]` or `[batch_size, max_time, ..., ...]` (if  `time_major` is set). May also be a nested `Tuple` with the same property. The first two dimensions *must* be the same across
all elements but all later dimensions may vary.
* `sequence_length`: Specifies length of each sequence in `inputs`.
* `initial_state`: A starting state for the RNN. If not provided, the initial state is `zero_state`.
* `dtype`: Data type of the initial state and output, in case it cannot be inferred.
* `parallel_iterations`: Number of iterations to run in parallel, defaulting to `32`. Ops that have no temporal dependency can be run in parallel and will be. Trades time for space - larger values use more memory.
* `swap_memory`: Optional `Bool`, which if `true` allows transparent swapping of `Tensor`s needed for back propagation from the CPU to the GPU and vice versa. Allows training of RNNs which wouldn't fit on a single GPU. Defaults to `false`.
* `time_major`: Shape format for `inputs` and `outputs` `Tensor`s. Determines whether the first dimension of each is `max_time` (`true`) or `batch_size` (`false`, default). `true` is more efficient but is the transpose of most TensorFlow operations.
* `scope`: `VariableScope` for the subgraph. Defaults to `RNN`.
"""
function dynamic_rnn(cell, inputs, sequence_length=nothing; initial_state=nothing, dtype=nothing, parallel_iterations=nothing, swap_memory=false, time_major=false, scope="RNN")
    input_dim = tf.get_shape(inputs, 3)
    #TODO Make this all work with non-3D inputs

    if time_major
        # TODO Do this in a more efficient way
        inputs=permutedims(inputs, [2,1,3])
    end

    num_steps = convert(tf.Tensor{Int64}, tf.shape(inputs)[2])
    if sequence_length === nothing
        # Works around a bug in upstream TensorFlow's while-loop
        # gradient calculation
        sequence_length = num_steps
    end


    initial_data = inputs[:,1,:]
    if initial_state === nothing
        initial_state = zero_state(cell, initial_data, dtype)
    end
    # By **MAGIC** these values end up in `while_output` even when num_steps=1
    
    # Calculate first output -- we can't trivially default it,
    # because that would require batch_size to be known statically,
    # and not having a fixed batch_size is pretty nice.
    output, state = cell(initial_data, initial_state, input_dim)
    # By **MAGIC** these values end up in `while_output` eve when num_steps=1
    # and the while-loop should not logically run
    
    time_step = tf.constant(2) #skip the completed first step
    while_output = @tf while time_step ≤ num_steps
        data = inputs[:, time_step, :]
        local new_state
        new_output = output


        tf.variable_scope(scope) do
            new_output, new_state = cell(data, state, input_dim)
            # Only update output and state for rows that are not yet passed their ends
            have_passed_end = sequence_length .< time_step
            f(old_arg, new_arg) = tf.select(have_passed_end, old_arg, new_arg)
            new_output = tf.struct_map(f, output, new_output)
            new_state = tf.struct_map(f, state, new_state)
        end

        [time_step=>time_step+1, state=>new_state, output=>new_output]
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
    tf.with_op_name(name, "Dropout") do
        keep_prob = convert(tf.Tensor{eltype(x)}, keep_prob)
        x_scaled = x/keep_prob
        if noise_shape == nothing
            noise_shape = tf.shape(x)
        end
        r = tf.random_uniform(noise_shape, 0, 1, seed=seed, dtype=eltype(x))
        y = x_scaled .* floor(keep_prob+r)
    end
    y
end

@op function sigmoid_cross_entropy_with_logits(;logits=nothing, targets=nothing, name=nothing)
    #  TODO make numerically stable
    local out
    tf.with_op_name(name, "SigmoidCrossEntropyWithLogits") do
        out = -logits.*targets + log(1+ exp(logits))
    end
    out
end


"""
`softmax_cross_entropy_with_logits(logits, labels, name=None)`

Computes softmax cross entropy between `logits` and `labels`.

Measures the probability error in discrete classification tasks in which the
classes are mutually exclusive (each entry is in exactly one class).  For
example, each CIFAR-10 image is labeled with one and only one label: an image
can be a dog or a truck, but not both.

**NOTE:**  While the classes are mutually exclusive, their probabilities
need not be.  All that is required is that each row of `labels` is
a valid probability distribution.  If they are not, the computation of the
gradient will be incorrect.

If using exclusive `labels` (wherein one and only
one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

**WARNING:** This op expects unscaled logits, since it performs a `softmax`
on `logits` internally for efficiency.  Do not call this op with the
output of `softmax`, as it will produce incorrect results.

`logits` and `labels` must have the same shape `[batch_size, num_classes]`
and the same dtype (either `float32` or `float64`).

##### Args:
*  <b>`logits`</b>: Unscaled log probabilities.
*  <b>`labels`</b>: Each row `labels[i]` must be a valid probability distribution.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
  softmax cross entropy loss.
"""
@op function softmax_cross_entropy_with_logits(;logits=nothing, labels=nothing, kwargs...)
    Ops.softmax_cross_entropy_with_logits(logits, labels; kwargs...)[1]
end

"""
    sparse_softmax_cross_entropy_with_logits(; labels=nothing, logits=nothing)

Computes sparse softmax cross entropy between `logits` and `labels`.

Measures the probability error in discrete classification tasks in which the
classes are mutually exclusive (each entry is in exactly one class).  For
example, each CIFAR-10 image is labeled with one and only one label: an image
can be a dog or a truck, but not both.

**NOTE:**  For this operation, the probability of a given label is considered
exclusive.  That is, soft classes are not allowed, and the `labels` vector
must provide a single specific index for the true class for each row of
`logits` (each minibatch entry).  For soft softmax classification with
a probability distribution for each entry, see
`softmax_cross_entropy_with_logits`.

**WARNING:** This op expects unscaled logits, since it performs a softmax
on `logits` internally for efficiency.  Do not call this op with the
output of `softmax`, as it will produce incorrect results.

A common use case is to have logits of shape `[batch_size, num_classes]` and
labels of shape `[batch_size]`. But higher dimensions are supported.

**Note that to avoid confusion, it is required to pass only named arguments to
this function.**

Args:
  labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
    `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
    must be an index in `[0, num_classes)`. Other values will raise an
    exception when this op is run on CPU, and return `NaN` for corresponding
    loss and gradient rows on GPU.
  logits: Unscaled log probabilities of shape
    `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float32` or `float64`.
  name: A name for the operation (optional).

Returns:
  A `Tensor` of the same shape as `labels` and of the same type as `logits`
  with the softmax cross entropy loss.
"""
@op function sparse_softmax_cross_entropy_with_logits(;logits=nothing, labels=nothing, name=nothing)
    Ops.sparse_softmax_cross_entropy_with_logits(logits, labels-1)[1]
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
    ids = tf.Tensor(ids)
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
@op function top_k(input, k=1; kwargs...)
    op = tf.get_op(Ops.top_kv_2(input, k; kwargs...))
    tf.Tensor(op, 1), tf.Tensor(op, 2)+1
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
    Ops.in_top_k(predictions, targets-1, k=k, name=name)
end

"""
`l2_loss(t)`

Computes half the L2-norm of a `Tensor` `t`, without taking the square root.
"""
@op function l2_loss(t; name=nothing)
    local out
    tf.with_op_name(name, "L2_Loss") do
        out = sqrt(reduce_sum(t.^2))
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

const local_response_normalization = Ops.lrn

@not_implemented function log_uniform_candidate_sampler()
end

@not_implemented function all_candidate_sampler()
end

@not_implemented function batch_norm_with_global_normalization()
end

@not_implemented function bias_add()
end

@not_implemented function fixed_unigram_candidate_sampler()
end

@not_implemented function conv_1d()
end

@not_implemented function conv_2d_transpose()
end

@not_implemented function atrous_conv_2d()
end

@not_implemented function depthwise_conv_2d()
end

@not_implemented function erosion_2d()
end

@not_implemented function weighted_cross_entropy_with_logits()
end

@op function l2_normalize(x, dim; epsilon=1e-12, name=nothing)
    # TODO take into account epsilon
    local out
    tf.with_op_name(name, "L2Normalize") do
        sums = tf.reduce_sum(x.^2, axis=[dim], keep_dims=true)
        norm = sqrt(sums)
        out = x/norm
    end
    out
end

include("seq2seq.jl")

## Deprecations

Base.@deprecate sigmoid_cross_entropy_with_logits(logits, targets; kwargs...) sigmoid_cross_entropy_with_logits(;logits=logits, targets=targets, kwargs...)

Base.@deprecate sparse_softmax_cross_entropy_with_logits(logits, labels; kwargs...) sparse_softmax_cross_entropy_with_logits(;logits=logits, labels=labels, kwargs...)

end
