"""
`function identity(input)`

Return a tensor with the same shape and contents as the input tensor or value.

Args:
  `input`: A `Tensor`.
  `name`: A name for the operation (optional).

Returns:
  A `Tensor`. Has the same type as `input`.
"""
function identity(input; name="")
    desc = NodeDescription("Identity", get_name(name))
    add_input(desc, Tensor(input))
    Tensor(Operation(desc))
end

function make_tuple(tensors; name="", control_inputs=Operation[])
    group_deps = group(vcat(tensors, control_inputs)...)
    ops = Tensor[]
    name_base = get_name(name)
    for (idx, input) in enumerate(tensors)
        n = string(name_base, "_", idx)
        desc = NodeDescription("Identity", n)
        add_input(desc, input)
        push!(ops, Tensor(Operation(desc)))
    end
    ops
end

"""
    `group(tensors...)`

Create an op that groups multiple operations.

When this op finishes, all ops in `input` have finished. This op has no
output.

See also `make_tuple` and `with_dependencies`.

Args:
  `inputs`: Zero or more tensors to group.
  `kwargs`: Optional parameters to pass when constructing the NodeDef.
  `name`: A name for this operation (optional).

Returns:
  An `Operation` that executes all its inputs.

Raises:
  `ValueError`: If an unknown keyword argument is provided.
"""
function group(tensors...; name="")
    desc = NodeDescription("NoOp", get_name(name))
    for tensor in tensors
        add_control_input(desc, tensor)
    end
    Tensor(Operation(desc))
end

"""
A named `Operation` that does nothing.
"""
function no_op(name="")
    desc = NodeDescription("NoOp", get_name(name))
    Tensor(Operation(desc))
end

"""
    count_up_to(ref, limit)

Increments `ref` until it reaches `limit`.

This operation outputs `ref` after the update is done.  This makes it
easier to chain operations that need to use the updated value.

Args:
*  `ref`: A mutable `Tensor`. Must be one of the following types: `Int32`, `Int64`.
    Should be from a scalar `Variable` node.
*  `limit`: An `int`.
    If incrementing `ref` would bring it above `limit`, instead generates an
    `OutOfRange` error.
*  `name`: A name for the operation (optional).

Returns:
*  A `Tensor`. Has the same type as `ref`.
*  A copy of the input before increment. If nothing else modifies the
   input, the values produced will all be distinct.
"""
function count_up_to(ref, limit; name="")
    desc = NodeDescription("CountUpTo", get_name(name))
    add_input(desc, Tensor(ref))
    desc["limit"] = Int64(limit)
    Tensor(Operation(desc))
end

"""
    cond(predicate::AbstractTensor, f1, f2)

Return either `fn1()` or `fn2()` based on the boolean predicate `pred`.

`fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
the same non-zero number and type of outputs.

Note that the conditional execution applies only to the operations defined in
`fn1` and `fn2`. Consider the following simple program:

```julia
z = Ka*btf.mul(a, b)
result = tf.cond(x .< y, ()-> x+z, ()-> square(y))
```

If `x` < `y`, the `tf.add` operation will be executed and `tf.square`
operation will not be executed. Since `z` is needed for at least one
branch of the `cond`, the `tf.mul` operation is always executed, unconditionally.
Although this behavior is consistent with the dataflow model of TensorFlow,
it has occasionally surprised some users who expected a lazier semantics.

Args:
*  `pred`: A scalar determining whether to return the result of `fn1` or `fn2`.
*  `fn1`: The callable to be performed if `pred` is `true`.
*  `fn2`: The callable to be performed if `pred` is `false`.
*  `name`: Optional name prefix for the returned tensors.

Returns:
*  `Tensor`s returned by the call to either `fn1` or `fn2`. If the callables
   return a singleton list, the element is extracted from the list.

Raises:
*  `TypeError`: if `fn1` or `fn2` is not callable.
*  `ValueError`: if `fn1` and `fn2` do not return the same number of tensors, or
              return tensors of different types.

Example:

```julia
  x = tf.constant(2)
  y = tf.constant(5)
  f1 = ()->17x
  f2 = ()->y+23
  r = cond(x.<y, f1, f2)
  # r is set to f1().
  # Operations in f2 (e.g., tf.add) are not executed.
```
"""
function Base.cond(pred::AbstractTensor, fn1, fn2; name="cond")
    #  TODO add control dependencies to subgraphs
    local switch1, switch2, merge

    with_op_name(name) do
        switch1 = NodeDescription("Switch", "switch1")
        add_input(switch1, fn1())
        add_input(switch1, Tensor(pred))
    end

    with_op_name(name) do
        switch2 = NodeDescription("Switch", "switch2")
        add_input(switch2, fn2())
        add_input(switch2, pred)
    end

    with_op_name(name) do
        merge = NodeDescription("Merge", "merge")
        add_input(merge, [Tensor(Operation(switch1), 2), Tensor(Operation(switch2), 1)])
    end
    Tensor(Operation(merge), 1)
end

@not_implemented function case()
end

@not_implemented function while_loop()
end
