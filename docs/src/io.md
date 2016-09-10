# Loading data with queues

TensorFlow.jl can load and preprocess data in parallel with training so that the performance of training is not limited by the speed of your IO device that holds the training data. The API is very similar to the Python TensorFlow API, so see [its docs](https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html).

Consult [the reference ](io_ref.md) for a list of all relevant methods.
