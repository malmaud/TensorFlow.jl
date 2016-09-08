function range_input_producer(limit; num_epochs=nothing, shuffle=true,
seed=nothing, capacity=32, name="")
    queue = FIFOQueue(capacity, Int)
    input = range(Tensor, 1; limit=limit)
    
end

@not_implemented function input_producer()
end

@not_implemented function slice_input_producer()
end

@not_implemented function string_input_producer()
end

@not_implemented function batch()
end

@not_implemented function batch_join()
end

@not_implemented function shuffle_batch()
end

@not_implemented function shuffle_batch_join()
end

type QueueRunner
    queue
    enqueue_ops
    QueueRunner() = new()
end

function QueueRunner(queue::AbstractQueue, enqueue_ops)
    self = QueueRunner()
    self.queue = queue
    self.enqueue_ops = enqueue_ops
    self
end

function create_threads(runner::QueueRunner, sess)
    for op in runner.enqueue_ops
        @schedule begin
            status = tf.Status()
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
        end
    end
end
