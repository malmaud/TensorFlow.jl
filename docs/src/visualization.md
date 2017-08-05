# Visualizing learning with Tensorboard

You can visualize your graph structure and various learning-related statistics using Google's Tensorboard tool. [Read its documentation](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html) to get a sense of how it works.

Write out summary statistics to a file using the `summary.FileWriter` type, which works in the same way as the Python version.

Generate the summaries using the summary operations, documented in [the reference](summary_ref.md). They incldue `summary.scalar`, `summary.histogram`, etc.


## Example

On the training side, your code will look like this

```julia
using TensorFlow
session = Session()

alpha = placeholder(Float32)
weights = Variable(...)
... # Set up the rest of your model

# Generate some summary operations
summary = TensorFlow.summary
alpha_summmary = summary.scalar("Learning rate", alpha)
weight_summary = summary.histogram("Parameters", weights)
merged_summary_op = summary.merge_all()

# Create a summary writer
summary_writer = summary.FileWriter("/my_log_dir")

# Train
for epoch in 1:num_epochs
  ... # Run training
  summaries = run(session, merged_summary_op)
  write(summary_writer, summaries, epoch)
end
```

Then from the console, run

```
> tensorboard --logdir=/my_log_dir
```
