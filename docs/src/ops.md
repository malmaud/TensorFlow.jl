# Operations

See the official TensorFlow documentation for a complete description of these operations.

## Basic operations

```@docs
placeholder
constant
concat
stack
split
expand_dims
argmin
argmax
add_n
one_hot
random_uniform
random_normal
```

## Variables

```@docs
Variable
global_variables_initializer
variable_scope
get_variable
ConstantInitializer
assign
assign_add
assign_sub
scatter_update
```

## Reductions

```@docs
reduce_sum
reduce_prod
reduce_min
reduce_max
reduce_all
reduce_any
reduce_mean
```

## Comparisons

```@docs
equal
not_equal
less
less_equal
greater
greater_equal
select
where
```

## Images

```@docs
image.decode_jpeg
image.encode_jpeg
image.decode_png
image.encode_png
image.resize_images
image.central_crop
```

## Neural networks

### Convolutions

```@docs
nn.conv2d
nn.max_pool
```

### Embeddings

```@docs
nn.embedding_lookup
```

### Recurrent neural nets

```@docs
nn.rnn
nn.dynamic_rnn
nn.zero_state
nn.output_size
nn.zero_state
```

### Nonlinearities

```@docs
nn.relu
nn.relu6
nn.elu
nn.softplus
nn.softsign
nn.softmax
nn.sigmoid
nn.tanh
```

### Losses

```@docs
nn.l2_loss
```

### Regularizations

```@docs
nn.dropout
```

### Evaluations

```@docs
nn.top_k
nn.in_top_k
```

## Logic

```@docs
logical_and
logical_not
logical_or
logical_xor
```

## Control flow

```@docs
identity
TensorFlow.make_tuple
TensorFlow.group
TensorFlow.no_op
cond
```
