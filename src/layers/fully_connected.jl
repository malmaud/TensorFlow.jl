"""
`fully_connected inputs,
num_outputs,
activation_fn=nn.relu,
normalizer_fn=nothing,
normalizer_params=nothing,
weights_initializer=initializers.xavier_initializer(),
weights_regularizer=nothing,
biases_initializer=init_ops.zeros_initializer,
biases_regularizer=nothing,
reuse=nothing,
variables_collections=nothing,
outputs_collections=nothing,
trainable=True,
scope=nothing)`

Adds a fully connected layer.

`fully_connected` creates a variable called `weights`, representing a fully
connected weight matrix, which is multiplied by the `inputs` to produce a
`Tensor` of hidden units. If a `normalizer_fn` is provided, such as
`batch_norm`, it is then applied. Otherwise, if `normalizer_fn` is
None and a `biases_initializer` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation_fn` is not `None`,
it is applied to the hidden units as well.
Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
prior to the initial matrix multiply by `weights`.
Args:
inputs: A tensor of with at least rank 2 and value for the last dimension,
i.e. `[batch_size, depth]`, `[-1, -1, -1, channels]`.
num_outputs: Integer or long, the number of output units in the layer.
activation_fn: activation function, set to `nothing` to skip it and maintain
a linear activation.
normalizer_fn: normalization function to use instead of `biases`. If
`normalizer_fn` is provided then `biases_initializer` and
`biases_regularizer` are ignored and `biases` are not created nor added.
default set to `nothing` for no normalizer function
normalizer_params: normalization function parameters.
weights_initializer: Not yet supported
weights_regularizer: Not yet supported
biases_initializer: Not yet supported
biases_regularizer: Not yet supported
reuse: whether or not the layer and its variables should be reused. To be
able to reuse the layer scope must be given.
variables_collections: Not yet supported
a dictionary containing a different list of collections per variable.
outputs_collections: Not yet supported
trainable: If `True` also add variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
scope: Not yet supported
Returns:
the tensor variable representing the result of the series of operations.

https://www.tensorflow.org/versions/r0.10/api_docs/python/contrib.layers.html#fully_connected
"""
function fully_connected(inputs,
    num_outputs;
    activation_fn=nn.relu,
    normalizer_fn=nothing,
    normalizer_params=nothing,
    reuse=false,
    trainable=true)

    weights_initializer = biases_initializer = Distributions.Normal(0,0.01)

    layer = variable_scope("fully_connected", reuse=reuse, initializer=weights_initializer) do
        inputs = Tensor(inputs)
        dtype = Float32
        inputs_shape = get_shape(inputs)
        num_input_units = inputs_shape.dims[end].value

        static_shape = inputs_shape.dims
        static_shape[end] = num_outputs

        out_shape = get_shape(inputs)
        out_shape.dims[end] = num_outputs

        weights_shape = [num_input_units, num_outputs]
        weights = get_variable("weights", weights_shape, dtype, trainable=trainable)
        length(static_shape) > 2 && (inputs = reshape(inputs, [-1, num_input_units]))

        outputs = inputs * weights
        if normalizer_fn != nothing
            normalizer_params = normalizer_params || Dict()
            outputs = normalizer_fn(outputs, normalizer_params)
        elseif biases_initializer != nothing
            biases = get_variable("biases",  [num_outputs], dtype, trainable=trainable)
            outputs = outputs + biases
        end
        activation_fn == nothing || (outputs = activation_fn(outputs))
        length(static_shape) > 2 && (outputs = reshape(outputs, pack(out_shape)))

        return outputs
    end

    return layer

end
