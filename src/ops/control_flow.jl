using Compat
import LinearAlgebra # cond

import .Ops: no_op, count_up_to

"""
     `identity(input)`

Return a tensor with the same shape and contents as the input tensor or value.
"""
@op Base.identity(tensor::AbstractTensor; name=nothing) = Ops.identity(tensor; name=name)

@op function make_tuple(tensors; name="", control_inputs=Operation[])
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
@op function group(tensors...; name=nothing)
    local desc
    with_op_name(name, "Group") do
        desc = NodeDescription("NoOp")
        for tensor in tensors
            add_control_input(desc, tensor)
        end
    end
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
@op function LinearAlgebra.cond(pred::AbstractTensor, fn1, fn2; name=nothing)
    #  TODO add control dependencies to subgraphs
    local merge
    with_op_name(name, "cond") do
        switch1 = Ops.switch(fn1(), pred)
        switch2 = Ops.switch(fn2(), pred)
        merge = Ops.merge([switch1[2], switch2[1]])
    end
    merge[1]
end

@not_implemented function case()
end

function with_frame(f, parallel_iterations, back_prop, swap_memory)
    op_context = get_def_graph().op_context
    if isempty(op_context.while_context)
        frame_name = "while"
    else
        frame_name = string(op_context.while_context[end].context_name, "/", "while")
    end
    while_context = tensorflow.WhileContextDef(parallel_iterations=parallel_iterations, back_prop=back_prop, swap_memory=swap_memory)
    setproperty!(while_context, :context_name, frame_name)
    add_to_collection(get_def_graph(), :while_context, while_context)
    setproperty!(while_context, :loop_exit_names, AbstractString[])
    setproperty!(while_context, :values_def, tensorflow.ValuesDef())
    setproperty!(while_context.values_def, :values, AbstractString[])
    push!(op_context.while_context, while_context)
    f()
    pop!(op_context.while_context)
end

"""
    while_loop(cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)


Repeat `body` while the condition `cond` is true.

`cond` is a callable returning a boolean scalar tensor. `body` is a callable
returning a (possibly nested) tuple, namedtuple or list of tensors of the same
arity (length and structure) and types as `loop_vars`. `loop_vars` is a
(possibly nested) tuple, namedtuple or list of tensors that is passed to both
`cond` and `body`. `cond` and `body` both take as many arguments as there are
`loop_vars`.

While `cond` evaluates to true, `body` is executed.

In addition to regular Tensors or IndexedSlices, the body may accept and
return TensorArray objects.  The flows of the TensorArray objects will
be appropriately forwarded between loops and during gradient calculations.

For correctness, `tf.while_loop()` strictly enforces shape invariants for
the loop variables. A shape invariant is a (possibly partial) shape that
is unchanged across the iterations of the loop. An error will be raised
if the shape of a loop variable after an iteration is determined to be more
general than or incompatible with its shape invariant. For example, a shape
of [11, None] is more general than a shape of [11, 17], and [11, 21] is not
compatible with [11, 17]. By default (if the argument `shape_invariants` is
not specified), it is assumed that the initial shape of each tensor in
`loop_vars` is the same in every iteration. The `shape_invariants` argument
allows the caller to specify a less specific shape invariant for each loop
variable, which is needed if the shape varies between iterations. The
[`Tensor.set_shape()`](../../api_docs/python/framework.md#Tensor.set_shape)
function may also be used in the `body` function to indicate that
the output loop variable has a particular shape. The shape invariant for
SparseTensor and IndexedSlices are treated specially as follows:

a) If a loop variable is a SparseTensor, the shape invariant must be
TensorShape([r]) where r is the rank of the dense tensor represented
by the sparse tensor. It means the shapes of the three tensors of the
SparseTensor are ([None], [None, r], [r]). NOTE: The shape invariant here
is the shape of the SparseTensor.shape property. It must be the shape of
a vector.

b) If a loop variable is an IndexedSlices, the shape invariant must be
a shape invariant of the values tensor of the IndexedSlices. It means
the shapes of the three tensors of the IndexedSlices are (shape, [shape[0]],
[shape.ndims]).

`while_loop` implements non-strict semantics, enabling multiple iterations
to run in parallel. The maximum number of parallel iterations can be
controlled by `parallel_iterations`, which gives users some control over
memory consumption and execution order. For correct programs, `while_loop`
should return the same result for any parallel_iterations > 0.

For training, TensorFlow remembers the tensors that are produced in the
forward inference but needed in back propagation. These tensors can be a
main source of memory consumption and often cause OOM problems when training
on GPUs.  When the flag swap_memory is true, we swap out these tensors from
GPU to CPU.  This for example allows us to train RNN models with very long
sequences and large batches.

Args:
  cond: A callable that represents the termination condition of the loop.
  body: A callable that represents the loop body.
  loop_vars: A (possibly nested) tuple, namedtuple or list of numpy array,
    `Tensor`, and `TensorArray` objects.
  shape_invariants: The shape invariants for the loop variables.
  parallel_iterations: The number of iterations allowed to run in parallel.
    It must be a positive integer.
  back_prop: Whether backprop is enabled for this while loop.
  swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
  name: Optional name prefix for the returned tensors.

Returns:
  The output tensors for the loop variables after the loop. When the length
  of `loop_vars` is 1 this is a Tensor, TensorArray or IndexedSlice and when
  the length of `loop_vars` is greater than 1 it returns a list.

Raises:
  TypeError: if `cond` or `body` is not callable.
  ValueError: if `loop_vars` is empty.

Example:

  ```python
  i = tf.constant(0)
  c = lambda i: tf.less(i, 10)
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])
  ```

Example with nesting and a namedtuple:

  ```python
  import collections
  Pair = collections.namedtuple('Pair', 'j, k')
  ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
  c = lambda i, p: i < 10
  b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
  ijk_final = tf.while_loop(c, b, ijk_0)
  ```

Example using shape_invariants:

  ```python
  i0 = tf.constant(0)
  m0 = tf.ones([2, 2])
  c = lambda i, m: i < 10
  b = lambda i, m: [i+1, tf.concat(0, [m, m])]
  tf.while_loop(
      c, b, loop_vars=[i0, m0],
      shape_invariants=[i0.get_shape(), tensor_shape.TensorShape([None, 2])])
  ```
"""
@op function while_loop(condition, body, variables; name=nothing, shape_invariants=nothing,
                        parallel_iterations=10, back_prop=true, swap_memory=false)
    error("Dynamic while loops are not currently ipmlemented.")
    #==
    g = Graph()
    def_graph = get_def_graph()
    # Not transfering def_graph.shapes as shape inference on placehoders is not used here.
    # Referentially linking everything else though.
    g.op_context = def_graph.op_context
    g.name_idx = def_graph.name_idx
    g.collections = def_graph.collections

    variable_tensors = Tensor.(get_tensors(variables))

    local output, g_def
    as_default(g) do
        with_op_name(name, "while") do
            with_frame(parallel_iterations, back_prop, swap_memory) do
                context = get_def_graph().op_context.while_context[end]
                @assert get_def_graph() === g


                # Preloop
                # prepare the Enter nodes, to later merge with their update values
                merge_nodes = Tensor[]
                enter_nodes = Tensor[]
                for var in variable_tensors
                    enter_op = Ops.enter(var, frame_name=context.context_name)
                    push!(enter_nodes, enter_op)
                    merge_op = Ops.merge([enter_op, enter_op])[1]
                        # For now set both inputs to be merged to be the same
                        # But later, we will reassign one of them to use the values from the body
                    push!(merge_nodes, merge_op)
                    fillin(merge_op.op)
                end
                setproperty!(context, :pivot_for_pred_name, get_name(merge_nodes[1]))

                # (Define the graph to)
                # evaluate the condition at each loop
                condition_out = with_op_control([tensor.op for tensor in merge_nodes]) do
                    merge_node_structs = build_output(variables, merge_nodes)
                    condition(merge_node_structs...)
                end
                pred = Ops.loop_cond(condition_out)
                setproperty!(context, :pivot_name, get_name(pred))

                # Body stuff
                # Define the output structure for the result of the body
                # For both the terminal and non-terminal loop increments
                output = Tensor[]
                body_input = Tensor[]
                for var_idx in eachindex(variable_tensors)
                    switch_false, switch_true = Ops.switch(merge_nodes[var_idx], pred)

                    exit_op = Ops.exit(switch_false)
                    push!(context.loop_exit_names, get_name(exit_op))
                    push!(output, exit_op)

                    body_pivot = identity(switch_true)
                    push!(body_input, body_pivot)
                end
                setproperty!(context, :pivot_for_body_name, get_name(body_input[1]))

                # (define the graph to)
                # actually execute the body
                body_output = with_op_control([tensor.op for tensor in body_input]) do
                    body_input_structs = build_output(variables, body_input)
                    body(body_input_structs...)
                end
                body_val = Ops.next_iteration.(get_tensors(body_output))


                # We are now complete with defining the loop structure,
                # in the graph `g`,
                # Now to manipulate that at the protobuf level
                # to link up all the plumbing in `def_graph`
                g_def = get_def(g)


                # Transfer the top-level values in the while loop body from the
                # while-loop graph to the main graph. Used when new variables
                # are defined inside the loop via `get_variable`.
                to_delete = Int[]
                for (op_idx, op) in enumerate(g_def.node)
                    for top_op in get_collection(:TopLevel)
                        if get_def(top_op).name == op.name
                            push!(to_delete, op_idx)
                        end
                    end
                end
                extend_graph(def_graph, g_def.node[to_delete])
                deleteat!(g_def.node, unique(to_delete))
                @assert get_collection(:TrainableVariables) ⊆  get_collection(:Variables)
                for var in get_collection(:Variables)
                    name = get_def(var.var_node).name
                    as_default(def_graph) do
                        Variable(name)
                        #NB: This does not discard information about collections
                        # as `g.collections` always has been a reference to `def_graph.collections`
                    end
                end


                #
                # Reassign one of the merge node inputs
                # to use the values variables from the body
                for var_idx in eachindex(variable_tensors)
                    for op in g_def.node
                        if op.name == merge_nodes[var_idx].op.name
                            # Recall earlier, we said we would reassign one of the inputs to the merge nodes?
                            # That we are doing now
                            v = body_val[var_idx]
                            op.input[2] = "$(get_def(v).name):$(v.value_index-1)"
                            #HACK: Doing this by string manipulation
                        end
                    end
                end

                for op in get_operations()
                    for output_idx in 1:num_outputs(op)
                        push!(context.values_def.values, get_name(Tensor(op, output_idx)))
                    end
                end
                setproperty!(context.values_def, :external_values, Dict{AbstractString, AbstractString}())
                for node in g_def.node
                    for (input_idx, input) in enumerate(node.input)
                        name, port = parse_port_name(input)
                        maybe_op = get_node_by_name(def_graph, name)

                        if (isnull(maybe_op)
                            || any(var.op.name == name && var.value_index == port for var in variable_tensors))

                            continue
                        end

                        # Failed to find input in variable_tensors
                        # and we did find an op by that name
                        # so that op must be the value we want
                        # Better process it.
                        op = get(maybe_op)
                        tensor = Tensor(op, port)
                        enter_op = as_default(def_graph) do
                            Ops.enter(tensor, frame_name=context.context_name, is_constant=true)
                        end
                        context.values_def.external_values[get_name(tensor)] = get_name(enter_op)
                        push!(context.values_def.values, get_name(tensor))
                        push!(context.values_def.values, get_name(enter_op))
                        node.input[input_idx] = get_name(enter_op)
                    end
                end
            end  # with_frame
        end  # with_op_name
    end  # as_default

    extend_graph(g_def.node)
    build_output(variables, output)
    ==#
end
