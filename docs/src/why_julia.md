# Why use the Julia TensorFlow API?

* Use Julia's JIT for fast ingestion of data, especially data in uncommon formats. With Python TensorFlow, you generally have to wait for the Python devs to create custom IO kernels in C.

* Use Julia for fast postprocessing of inference results, such as calculating various statistics and visualizations which don't have a canned vectorized implementation.

* Use Julia's multiple dispatch to make it easy to specify models with native-looking Julia code. For example, native functions and operations like `sin`, `*` (matrix multiplication), `.*` (element-wise multiplication), etc work just as well on tensors as they do on native Julia values. Compare to Python, which requires learning specialized namespaced functions like `tf.matmul`.

* Use Julia metaprogramming to simplify graph construction that requires code repetition in Python. For example, creating a named tensor in Python often takes the form `i = tf.constant(1, name="i")`. In Julia, you can just write `@tf i = constant(1)` for the same effect.

* Another example is while loops. Consider taking the sum of the square of the first ten integers. In Python, this is

```python
i = tf.constant(0, name="i")
result = tf.constant(0, name="result")
output = tf.while_loop(lambda i, result: tf.less(i, 10), lambda i, result: [i+1, result+tf.pow(i,2), [i, result])
```

while in Julia, you can use native loops:

```julia
@tf i = constant(0)
@tf result = constant(0)
output = @tf while i < 10
  i_sq = i^2
  [i=>i+1, result=>result+i_sq]
end
```
