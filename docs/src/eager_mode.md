# Eager execution

TensorFlow.jl supports the more recent eager execution mode supported by the Python TensorFlow client. See the [Python docs](https://www.tensorflow.org/guide/eager) for an overview of the functionality. TensorFlow.jl tries to mirror the Python API for eager execution.

In short, instead of creating a `Session` object and calling `run`, the value of expressions involving TensorFlow tensors is computed immediately:

```julia
using TensorFlow
enable_eager_execution()
x = constant(1)
y = 2
z = x + y
z
```

produces 

```
EagerTensor<Ptr{Nothing} @0x000000011a0760a0>(3)
```

Note the '3', showing the result of the computation even though `run` was never explicitly called. 

Compare this to the more verbose non-eager form of this example:

```julia
using TensorFlow
session = Session()
x = constant(1)
y = 2
z = x+ y
z_value = run(session, z)
```

Eager mode works well with the new Julia debugger - just set a breakpoint in the middle of your computation and you can freely inspect the values of all the intermediate TensorFlow calculations.

## How to enable
Call `enable_eager_exeuction` before using any other TensorFlow methods. Once enabled in a Julia session, it cannot be disabled.

## Computing gradients

In eager mode, gradients require using a tape to record computations. Tapes are created by calling `create_tape()`, which returns a new tape and (by default) sets that tape to be the active tape used by all subsequent operations. 

A complete example:

```julia
using TensorFlow
tf = TensorFlow
tape = create_tape()
x = constant(3.0)
y = 2x
z = log(y)
z_grad = tf.grad(tape, z, x)
```

`z_grad` will hold the gradient of `z` with respect to `x`. Since `z=log(2x)`, we know analytically that `dz/dx = 1/x`, and indeed `z_grad` is `1/3.0`. 

To save memory, the tape is cleared after being used in a gradient calculation. Thus after running the above code, `grad(tape, y, x)` would not work as the tape has already "forgotten" how `y` depends on `x`.

To override the tape used, the `with_tape` function with a `do` block can be used:

```julia
tape1 = create_tape()
tape2 = create_tape()  # Normally tape2 would now be the active tape
local x
local y
with_tape(tape1) do
    x = constant(1.0)
    y = 2x  # But tape1 is going to be used to record this operation
end

y_grad = grad(tape1, y, x)

```


###  Caveats

Gradients are computed purely in Julia in eager mode, which requires reimplementing the gradient of all TensorFlow operations in Julia. In non-eager model, gradients are computed by the Google-maintained TensorFlow C library. As a result, there is a greater chance that there is a bug or unimplemented gradient in eager mode. 

If an operation that you are using doesn't have a gradient implemented for it currently, it is easy register a backwards function that implements one. See the calls to `@back_for` in `tape.jl`. Pull requests that implement gradients for additional operators are very welcome.
