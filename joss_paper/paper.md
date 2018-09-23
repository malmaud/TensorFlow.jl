---
title: 'TensorFlow.jl: An idiomatic Julia wrapper for TensorFlow '
tags:
  - julialang
  - tensorflow
  - machine learning
  - neural networks
  - deep learning
authors:
 - name: Jonathan Malmaud
   affiliation: 1
 - name: Lyndon White
   orcid: 0000-0003-1386-1646
   affiliation: 2

affiliations:
 - name: Massachusetts Institute of Technology
   index: 1
 - name: The University of Western Australia
   index: 2

date: 23 Sep 2018
bibliography: paper.bib
---

# Summary
TensorFlow.jl is the Julia ([@julia2014]) client library for the TensorFlow deep-learning framework ([@tensorflow2015],[@tensorflow2016]).
This is accomplished by defining methods which define tensorflow graphs;
which can then be executed to perform training or inference.
The methods are defined by overloading functions to operate on a `Tensor` type, which represents a node in the graph.
In doing so, a lot of standard julia functionality can be made to work.
Including often allowing code that is unaware of TensorFlow manipulate the compuational graph via _duck-typing_.
TensorFlow.jl has elegant syntax.
It allows all the usual infix functions such as `+`, `-`, `.*` etc.
Further it allows Julia-style indexing (e.g. `x[:, ii + end÷2]), and concentration (e.g. `[A B]`, `[x; y; 1]`).
It's goal is to be idiomatic for Julia users,
while still preserving all the power and maturity of the TensorFlow system.



There exists the need to carefully balance between matching the Python TensorFlow API, and Julia conventions.
The Python client is itself designed to closely mirror numpy.
Julia follows different conventions.
TensorFlow.jl attempts to strike a balance between the two.
Some examples are shown in the table below.

| **Julia**                                                                                                           | **Python TensorFlow**                                                                                                                   |  **Julia TensorFlow.jl**                                                                                                                                                                                                                 |
|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1-based indexing                                                                                                    |  0-based indexing                                                                                                                       | 1-based indexing                                                                                                                                                                                                                         |
| explicit broadcasting                                                                                               |  implicit broadcasting                                                                                                                  |  implicit or explicit broadcasting                                                                                                                                                                                                       |
| last index at `end` 2nd last in `end-1`                                                                             |  last index at `-1` second last in `-2`                                                                                                 | last index at `end` 2nd last in `end-1`                                                                                                                                                                                                  |
| Operations in Julia ecosystem namespaces. (`SVD` in `LinearAlgebra`, `erfc` in `SpecialFunctions`, `cos` in `Base`) |  All operations TensorFlow's namespaces  (`SVD` in `tf.linalg`, `erfc` in `tf.math`,  `cos` in `tf.math`, and all reexported from `tf`) |  All hand imported Operations in the Julia ecosystems namespaces. (`SVD` in `LinearAlgebra`, `erfc` in `SpecialFunctions`, `cos` in `Base`) Ops that have no other place are in `TensorFlow`. Automatically generated ops are in `Ops`   |
| Column Major                                                                                                        |  Row Major                                                                                                                              |  Row Major                                                                                                                                                                                                                               |
| Container types are parametrized by number of dimensions and element type                                           |  N/A: does not have a parametric type system                                                                                            | Tensors are parametrized by element type                                                                                                                                                                                                 |



With some introspection, it can be noted that the graph definition of TensorFlow, is itself meta-programming.
More generally, it has been noted that frameworks such as TensorFlow are in fact complete programming languages embedded inside another language (@[MLandPL]).
This often comes with some awkwardness, as the syntax and the semantics of the embedded language, by definition do not match the host language or there would be no need for the former's existence.
Julia is a significantly more flexible language than Python,
and though this has some downsides,
it allows us to significantly reduce this impedance of working with an embedded language like TensorFlow.

One example of our ability to leverage the increased expressiveness of Julia is using `@tf` macro blocks to automatically name nodes.
Nodes in a TensorFlow graph have names; these correspond to variable names in a traditional programming language.
Thus every operation, variable and placeholder takes a `name` parameter.
In most TensorFlow bindings, these must be specified manually resulting in a lot of code that includes duplicate information such as
`x = tf.placeholder(tf.float32, name="x")` or they are defaulted to an uninformative value such as `Placeholder_1`.
In TensorFlow.jl, prefixing a lexical block (such as a `function` or a `begin` block) with the `@tf` macro,
will cause the `name` parameter on all operations occurring on the right-hand side of an assignment to be filled in using the left-hand side.
That is to say nodes in the computational graph, that are assigned to variables will automatically be given the name matching the variable name.
This makes for more intuitive debugging, and graph visualisation.

Another example of the use of Julia's metaprogramming is in the automatic generation of the binding code.
The C API can be queried to return protobuf definitions of all operations.
This described the operations at a sufficient level to generate the Julia code to bind to the functions in the C API.
One challenge in this is that such generated code, must correct the indices to be 1 based, as per Julia convention.
This is accomplished using a set of rules based on the conventions used in the definition for naming variables.
Using this automatic generation, any of the operations defined in the C API can be easily imported into Julia.
The most commonly used are used to generate code for the Ops module, and generally reexported while bound to Julia functions.
More obscure operation can be imported at runtime using `@tfimport OperationName`, which will generate the binding and load it immediately.



TensorFlow.jl represents all nodes in the computational graph as `Tensors` which are parametrised by their element type, e.g. `Int`, `Float64` or `Bool`.
This allows Julia's dispatch system to be used during the construction of the graph.
For example in the definition of indexing operations,
this dispatch is used to transform code such as `x[b]`, `x[:, end-ii]`, or `x[ii:jj, :, :]`, into appropriate combinations of `gather_nd`, `slice`, `boolean_mask` etc depending upon the type of the indexes.
It is also used to cast inputs in various functions to compatible shapes.



## Challenges

A significant difficulty in implementing the TensorFlow.jl package for julia,
is that in the TensorFlow v0 and v1, the C API is designed for the execution of pretrained models;
and not for the definition or training of graphs.
The C-API primarily exposes low-level operations such as `matmul` or `reduce_sum`.
Optimizers, RNNs functionality and until recently shape-inference all required reimplementation.
Most challengingly, the symbolic differentiation implemented in the `gradients` function is not available from the C API.
To work around this, we currently use [Python interop](https://github.com/JuliaPy/PyCall.jl), to generate the gradient nodes using the Python client.

This has been improving over time, both with more aspects moving into the C API,
 and with more complete coverage of other aspects from our own efforts.
There never-the-less remains a large number of components from the upstream contrib submodule, that remain unimplemented.



## Other deep learning frameworks in Julia

Julia also has bespoke neural network packages such as Mocha ([@mocha2014]),  Knet ([@knet2016) and Flux ([@flux2018]).
As well as bindings to other frameworks such as MxNet ([@mxnet2015]).
While not having the full-capacity to directly leverage some of the benefits of the language and its ecosystem present in the pure julia frameworks such as Flux,
TensorFlow.jl provides an interface to one of the most mature and widely deployed deep learning environments.
It thus trivially intrinsically supports technologies such as TPUs, and systems such as TensorBoard.
It gains benefits from the any optimisations made in the graph execution engine.


## Acknowledgements



# References
