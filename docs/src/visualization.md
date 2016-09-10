# Visualizing learning with Tensorboard

You can visualize your graph structure and various learning-related statistics using Google's Tensorboard tool. [Read its documentation](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html) to get a sense of how it works. Note that TensorFlow.jl does *not* come with Tensorboard - it comes with the Python TensorFlow package.

Write out summary statistics to a file using the `SummaryWriter` type, which works in the same way as the Python version. Generate the summaries using the summary operations:

```@docs
scalar_summary
histogram_summary
image_summary
merge_summary
merge_all_summaries
```

## Example

On the training side, your code will look like this

```julia
session = Session()

alpha = placeholder(Float32)
weights = Variable(...)
... # Set up the rest of your model

# Generate some summary operations
alpha_summmary = scalar_summary("Learning rate", alpha)
weight_summary = histogram_summary("Parameters", weights)
merged_summary_op = merge_all_summaries()

# Create a summary writer
summary_writer = train.SummaryWriter("/my_log_dir")

# Train
for epoch in 1:num_epochs
  ... # Run training
  summaries = run(session, merged_summary_op)
  write(summary_writer, summaries, epoch)
end
```

Then from the console, run

```
> tensorboard --log_dir=/my_log_dir
```
