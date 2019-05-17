
TensorFlow v0.12 Release Notes
================================

* The upstream TensorFlow version has been bumped to 1.13.1.


TensorFlow v0.11 Release Notes
==============================

* Support for eager execution mode has been added. See [the docs](http://malmaud.github.io/TensorFlow.jl/latest/eager_mode.html) for instructions on how to use it.


TensorFlow v0.10.0 Release Notes
==============================

* Support for Julia 1.0 has been added. Support for all prior versions of Julia (except 0.7, which is functionally identical to 1.0) has been dropped.

TensorFlow v0.7 Release Notes
=============================

* Support for Julia 0.5 has been dropped.
* Enhanced support for visualization with TensorBoard.
* Operations defined in C are now accessed by `import_op(<op name>)` instead of `Ops.<op name>`.

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
