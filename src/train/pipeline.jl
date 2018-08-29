"""
`add_queue_runner(runner::QueueRunner)`

Manually adds a queue runner to the graph's collection of queue runners.
All runners in the collection will be executed in parallel when `start_queue_runners` is invoked.
"""
function add_queue_runner(runner, collection=:QueueRunners)
    tf.add_to_collection(get_def_graph(), collection, runner)
end

"""
`start_queue_runners(session::Session)`

Run all queue runners in parallel that were previously added to the graph's collection of queue runners
via `add_queue_runner`.

Args:
* `session`: The TensorFlow session containing the queues
"""
function start_queue_runners(sess)
    runners = tf.get_collection(:QueueRunners)
    for runner in runners
        create_threads(runner, sess)
    end
end

"""
`clear_queue_runners()`

Remove all queue runners from the queue.
"""
function clear_queue_runners()
    g = tf.get_def_graph()
    empty!(g.collections[:QueueRunners])
end

"""
`range_input_producer(limit; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32)`

Produces the integers from 1 to `limit` in a queue.

Args:
* `limit`: Inclusive upper bound on the endpoint of the range of integers to produce
* `num_epochs`: Number of times to produce the integers.
* `do_shuffle`: If `true`, shuffle the inputs each epoch.
* `seed`: Seed to use for the RNG if `do_shuffle` is `true`.
* `capacity`: Sets the queue capacity. Default is 32.
"""
@op function range_input_producer(limit; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name=nothing)
    local out
    with_op_name(name, "RangeInputProducer") do
        input = range(Tensor, 1, limit=limit+1)
        input_producer(input, element_shape=[], num_epochs=num_epochs, do_shuffle=do_shuffle, seed=seed, capacity=capacity, name=name)
    end
    out
end

"""
`input_producer(input; element_shape=nothing, num_epochs=nothing, do_shuffle=true, seed=0, capacity=32)`

Outputs the rows of `input` to a queue for input pipelining.

Args:
* `input`: A `Tensor` with the rows to produce.
* `element_shape`: The shape of the input rows, in case it can't be inferred. Defaults to `nothing`.
* `num_epochs`: Number of times to produce each row. If unspecified (default), `input_producer` can produce each row an unlimited number of times.
* `do_shuffle`: If `true`, shuffle the inputs each epoch.
* `seed`: Seed to use for the RNG if `do_shuffle` is `true`.
* `capacity`: Sets the queue capacity. Default is 32.
"""
@op function input_producer(input; element_shape=nothing, num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name=nothing)
    local queue
    with_op_name(name, "InputProducer") do
        input = convert(Tensor, input)
        if element_shape === nothing
            queue = tf.FIFOQueue(capacity, eltype(input))
        else
            queue = tf.FIFOQueue(capacity, eltype(input), shapes=[element_shape])
        end
        if do_shuffle
            input = shuffle(input, seed=seed)
        end
        if num_epochs !== nothing
            epochs = tf.Variable(1, name="num_epochs", trainable=false)
            epoch_op = tf.count_up_to(epochs, num_epochs+1)
        else
            epoch_op = tf.no_op()
        end
        enqueue_op = tf.enqueue_many(queue, input)
        op = tf.group(enqueue_op, epoch_op)
        runner = QueueRunner(queue, [op])
        add_queue_runner(runner)
    end
    return queue
end

"""
`slice_input_producer(input; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32)`

Produces a slice of each `Tensor` in `input`.

Args:
* `input`: A list of `Tensor` objects. Each element must have the same size of its first dimension.
* `num_epochs`: Number of times to produce the strings.
* `do_shuffle`: If `true`, shuffle the inputs each epoch.
* `seed`: Seed to use for the RNG if `do_shuffle` is `true`.
* `capacity`: Sets the queue capacity. Default is 32.
"""
@op function slice_input_producer(input; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name=nothing)
    input_producer(convert(Tensor, input), num_epochs=num_epochs, do_shuffle=do_shuffle, seed=seed, capacity=capacity, name=name)
end

"""
`string_input_producer(string_tensors; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32)`

Output strings to a queue for an input pipeline.

Args:
* `string_tensor`: A one dimensional `Tensor` of strings to produce.
* `num_epochs`: Number of times to produce the strings.
* `do_shuffle`: If `true`, shuffle the inputs each epoch.
* `seed`: Seed to use for the RNG if `do_shuffle` is `true`.
* `capacity`: Sets the queue capacity. Default is 32.
"""
@op function string_input_producer(string_tensor; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name=nothing)
    input_producer(convert(Tensor,string_tensor), num_epochs=num_epochs, do_shuffle=do_shuffle, seed=seed, capacity=capacity, name=name)
end

"""
`shuffle_batch(tensors, batch_size; capacity=32, enqueue_many=false, shapes=nothing, dynamic_pad=false, allow_smaller_final_batch=false)`

Create batches by randomly shuffling `tensors`.

Args:
* `tensors`: A list of tensors to enqueue.
* `batch_size`: The batch size which will be pulled from the queue.
* `capacity`: Sets the queue capacity. Default is 32.
* `dynamic_pad`: If `true` all `tensors` will be padded on unknown dimensions to `batch_size`. Otherwise `tensors` shapes must be fully known. Currently only `false` is supported.
* `enqueue_many`: If `false`, `tensors` represents a single example. Otherwise `tensors` represents a batch of examples. Currently only `false` is supported.
* `shapes`: The shapes for each example. Defaults to the inferred shapes from `tensors`.
* `allow_smaller_final_batch`: If `true` (default `false`), the final batch is allowed to be smaller than the other batches if there are not enough samples remaining.
"""
@op function shuffle_batch(tensors, batch_size; capacity=32, enqueue_many=false, shapes=nothing, allow_smaller_final_batch=false, name=nothing)
    local dequeue_op
    with_op_name(name, "ShuffleBatch") do
        if enqueue_many || dynamic_pad
            error("Not supported")  # TODO support this
        end
        if shapes === nothing
            shapes = [tf.get_shape(x) for x in tensors]
        end
        queue = tf.RandomShuffleQueue(capacity, [eltype(x) for x in tensors], shapes=shapes, name="queue")
        enqueue_op = tf.enqueue(queue, tensors, name="enqueue")
        dequeue_op = tf.dequeue_many(queue, batch_size, name="dequeue")
        runner = QueueRunner(queue, [enqueue_op])
        add_queue_runner(runner)
    end
    return dequeue_op
end

@not_implemented function batch_join(tensors, batch_size; capacity=32, enqueue_many=false, shapes=nothing, dynamic_pad=false, allow_smaller_final_batch=false, name="BatchJoin")
end

"""
`batch(tensors, batch_size; num_threads=1, capacity=32, enqueue_many=false, shapes=nothing, dynamic_pad=false, allow_smaller_final_batch=false)`

Create batches by randomly shuffling `tensors`.

Args:
* `tensors`: A list of tensors to enqueue.
* `batch_size`: The batch size which will be pulled from the queue.
* `num_threads`: The number of threads to use while enqueuing. Default is 1.
* `capacity`: Sets the queue capacity. Default is 32.

* `dynamic_pad`: If `true` all `tensors` will be padded on unknown dimensions to `batch_size`. Otherwise `tensors` shapes must be fully known. Currently only `false` is supported.
* `enqueue_many`: If `false`, `tensors` represents a single example. Otherwise `tensors` represents a batch of examples. Currently only `false` is supported.
* `shapes`: The shapes for each example. Defaults to the inferred shapes from `tensors`.
* `allow_smaller_final_batch`: If `true` (default `false`), the final batch is allowed to be smaller than the other batches if there are not enough samples remaining.
"""
@op function batch(tensors, batch_size; num_threads=1, capacity=32, enqueue_many=false, shapes=nothing, dynamic_pad=false, allow_smaller_final_batch=false, name=nothing)
    local dequeue_op
    with_op_name(name, "Batch") do
        if enqueue_many || dynamic_pad
            error("Not supported")  # TODO support this
        end
        if shapes === nothing
            shapes = [tf.get_shape(x) for x in tensors]
        end
        queue = tf.FIFOQueue(capacity, [eltype(x) for x in tensors], shapes=shapes, name="queue")
        enqueue_op = tf.enqueue(queue, tensors, name="enqueue")
        dequeue_op = tf.dequeue_many(queue, batch_size, name="dequeue")
        runner = QueueRunner(queue, [enqueue_op])
        add_queue_runner(runner)
    end
    return dequeue_op
end

@not_implemented function shuffle_batch_join()
end

"""
`QueueRunner`

Represents an object that continually enqueues elements to a TensorFlow queue in parallel with other operations.
"""
mutable struct QueueRunner
    queue
    enqueue_ops
    QueueRunner() = new()
end

"""
`QueueRunner(queue::AbstractQueue, enqueue_ops)`

Constructs a `QueueRunner`.

Args:
* queue: The queue which is being enqueued to
* enqueue_ops: A list of operations that enqueue tensors to `queue`. These operations are allowed to have side effects.
"""
function QueueRunner(queue::AbstractQueue, enqueue_ops)
    self = QueueRunner()
    self.queue = queue
    self.enqueue_ops = enqueue_ops
    self
end

"""
`create_threads(runner::QueueRunner, session::Session)`

Creates tasks that run the enqueue operations in `runner` in parallel.
"""
function create_threads(runner::QueueRunner, sess)
    tasks = Task[]
    for op in runner.enqueue_ops
        task = @async begin
            status = tf.Status()
            while true
                try
                    @threadcall((:TF_SessionRun, tf.LIBTF), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Ptr{Cvoid}),
                        sess.ptr,
                        C_NULL,
                        C_NULL,
                        C_NULL,
                        0,
                        C_NULL,
                        C_NULL,
                        0,
                        [tf.get_op(op).ptr],
                        1,
                        C_NULL,
                        status.ptr)
                    tf.check_status(status)
                catch err
                    @info("got $err on queue")
                    break
                end
            end
        end
        push!(tasks, task)
    end
    return tasks
end
