# Operations

See the official TensorFlow documentation for a complete description of these operations.

## Basic operations

```@docs
placeholder
constant
concat
pack
split
expand_dims
argmin
one_hot
random_uniform
```

## Variables

```@docs
Variable
initialize_all_variables
variable_scope
get_variable
ConstantInitializer
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
image.decode_png
image.resize_images
```

## Neural networks

### Convolutions

```@docs
nn.conv2d
nn.max_pool
```

### Embeddings

```@docs
embedding_lookup
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
make_tuple
group
no_op
cond
```

## Tensorboard summaries

```@docs
scalar_summary
```
