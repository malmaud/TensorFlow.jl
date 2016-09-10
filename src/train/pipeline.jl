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
* session: The TensorFlow session containing the queues
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
* limit: Inclusive upper bound on the endpoint of the range of integers to produce
* num_epochs: Number of times to produce the integers.
* do_shuffle
* seed
* capacity
"""
function range_input_producer(limit; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name="")
    input = range(Tensor, 1, limit=limit+1)
    input_producer(input, element_shape=[], num_epochs=num_epochs, do_shuffle=do_shuffle, seed=seed, capacity=capacity, name=name)
end

"""
`input_producer(input; element_shape=nothing, num_epochs=nothing, do_shuffle=true, seed=0, capacity=32)`
"""
function input_producer(input; element_shape=nothing, num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name="")
    full_name = tf.get_name(name)
    input = tf.to_tensor(input)
    if element_shape === nothing
        queue = tf.FIFOQueue(capacity, eltype(input))
    else
        queue = tf.FIFOQueue(capacity, eltype(input), shapes=[element_shape])
    end
    if do_shuffle
        input = shuffle(input, seed=seed)
    end
    if num_epochs !== nothing
        epochs = tf.Variable(1, name="$full_name/num_epochs", trainable=false)
        epoch_op = tf.count_up_to(epochs, num_epochs+1)
    else
        epoch_op = tf.no_op()
    end
    enqueue_op = tf.enqueue_many(queue, input)
    op = tf.group(enqueue_op, epoch_op)
    runner = QueueRunner(queue, [op])
    add_queue_runner(runner)
    return queue
end

@not_implemented function slice_input_producer()
end

"""
`string_input_producer(string_tensors; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32)`
"""
function string_input_producer(string_tensor; num_epochs=nothing, do_shuffle=true, seed=0, capacity=32, name="")
    input_producer(tf.to_tensor(string_tensor), num_epochs=num_epochs, do_shuffle=do_shuffle, seed=seed, capacity=capacity, name=name)
end

"""
`shuffle_batch(tensors, batch_size; capacity=32, enqueue_many=false, shapes=nothing, dynamic_pad=false, allow_smaller_final_batch=false)`
"""
function shuffle_batch(tensors, batch_size; capacity=32, enqueue_many=false, shapes=nothing, dynamic_pad=false, allow_smaller_final_batch=false, name="")
    if enqueue_many
        error("Not supported")  # TODO support this
    end
    name = tf.get_name(name)
    if shapes === nothing
        shapes = [tf.get_shape(_) for _ in tensors]
    end
    queue = tf.RandomShuffleQueue(capacity, [eltype(_) for _ in tensors], shapes=shapes, name="$name/queue")
    enqueue_op = tf.enqueue(queue, tensors, name="$name/enqueue")
    dequeue_op = tf.dequeue_many(queue, batch_size, name="$name/dequeue")
    runner = QueueRunner(queue, [enqueue_op])
    add_queue_runner(runner)
    return dequeue_op
end

@not_implemented function batch_join()
end

@not_implemented function batch()

end

@not_implemented function shuffle_batch_join()
end

"""
`QueueRunner`

Represents an object that continually enqueues elements to a TensorFlow queue in parallel with other operations.
"""
type QueueRunner
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
        task = @schedule begin
            status = tf.Status()
            while true
                try
                    @threadcall(:TF_SessionRun, Void, (Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Ptr{Ptr{Void}}, Cint, Ptr{Void}, Cint, Ptr{Void}, Ptr{Void}),
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
                    info("got $err on queue")
                    break
                end
            end
        end
        push!(tasks, task)
    end
    return tasks
end
