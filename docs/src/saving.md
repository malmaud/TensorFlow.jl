# Saving and restoring network graphs and variables

Both the value of variables (typically weights in a neural network) and the topology of the network (ie, all the operations that constitute your model) can be serialized to disk and loaded later. The mechanism is different for both.

## Saving and restoring variable values

See the [main TensorFlow docs](https://www.tensorflow.org/programmers_guide/variables#saving_variables) for an overview.

First create a saver object:

```julia
saver = train.Saver()
```

Then call the `save` method to save your variables:

```julia
train.save(saver, session, "/path/to/variable_file")
```

The newly-created "variable_file" is a [JLD2](https://github.com/simonster/JLD2.jl) file that contains a mapping from variable names to their values. The value of variables can later be restored in a new Julia session with

```julia
train.restore(saver, session, "/path/to/variable_file")
```

For example, in one Julia session I might have

```julia
using TensorFlow
session = Session()
@tf x = get_variable([], Float32)
run(session, assign(x, 5.0f0))
saver = train.Saver()
train.save(saver, session, "weights.jld")
```

Then to restore in another session,

```julia
using TensorFlow
session = Session()
@tf x = get_variable([], Float32)
saver = train.Saver()
train.restore(saver, session, "weights.jld")
run(session, x)  # Outputs '5.0f0'
```


Just as in the Python API, there is an easy way to save multiple variable files during different iterations of training, making it easy to retrospectively analyze the value of variables during training.

 `save` can be passed a `global_step` keyword parameter, which is an integer that will be suffixed to the variable file name. The `Saver` constructor accepts an optional `max_to_keep` argument, which is an integer specifying how many of the latest versions of the variable files to save (older ones will be discarded to save space). For example, this code will keep the value of variables during the 5 most recent training iterations:

 ```julia
 ...
 saver = train.Saver(max_to_keep=5)
 for iteration in 1:100
   ...
   train.save(saver, session, "variable_file", global_step=iteration)
end
```

By the end of this loop, file "variable_file_95" contains the variable values during the 95th iteration, "variable_file_96" the 96th iteration, etc.

## Saving and restoring models

The actual structure of the model can also be saved and restored from disk. In TensorFlow jargon, the complete structure of the model is referred to as the "metagraph".

To save the metagraph, call

```julia
train.export_meta_graph("filename")
```

To restore it, call

```julia
train.import_meta_graph("filename")
```

For example, in one Julia session you might write

```julia
using TensorFlow
x = constant(1)
@tf y = x+1
train.export_meta_graph("my_model")
```

Then in a new Julia session,
```julia
using TensorFlow
session = Session()
train.import_meta_graph("my_model")
y = get_tensor_by_name("y")
run(session, y)  # Outputs '2'
```


The metagraph file format is the same as that used by the Python TensorFlow version, so models can be freely passed to and from Python TensorFlow sessions.
