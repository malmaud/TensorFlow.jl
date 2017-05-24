TensorFlow v0.6 Release Notes
=============================

API deprecations
------------

The API has changed to resemble the finalized TensorFlow 1.0 API. See [the TensorFlow release notes](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md) for a list of the changes.

In particular for TensorFlow.jl:

* The summary operations, like `train.scalary_summary`, have moved to `summary.scalar`.
* `train.SummaryWriter` has moved to `summary.FileWriter`.
* The `reduction_indices` keyword argument have changed to `axis`in every function which used to take a `reduction_indices` argument.
* `mul` has changed to `multiply`, and likewise for `sub` and `neg`.
* `pack` is now `stack` and `unpack` is now `unstack`.
* The `*_cross_entropy_with_logits` family of functions has changed to only accept keyword arguments instead of positional arguments.
* The order of arguments to `concat` has switched (tensors to concatenate come first, followed by the concetenation axis).


Highlights
----------------

* `dynamic_rnn` has been added
* `while_loop` has been added, with a convenient `@tf while ... end` syntax
* Support for Docker via official Docker images

Switch to 1-based indexing
---------------

Operations which take indices as arguments now expect the arguments to be
1-based instead of 0-based, which was a hold-over of TensorFlow's Python
legacy. This affects the following functions:

* The `axis` parameter for any operation which takes an `axis` parameter
* The `begin_` argument of `slice`

All TensorFlow operations now supported
--------------

Every operation defined by TensorFlow is now automatically wrapped in a
Julia function available in the `Ops` module.
