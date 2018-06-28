abstract type AbstractQueue end

mutable struct FIFOQueue <: AbstractQueue
    op::Operation
    dtypes
    FIFOQueue() = new()
end

function Base.show(io::IO, f::FIFOQueue)
    print(io, "FIFOQueue")
end

mutable struct RandomShuffleQueue <: AbstractQueue
    op::Operation
    dtypes
    RandomShuffleQueue() = new()
end

@op function FIFOQueue(capacity, dtypes; name=nothing, shapes=nothing)
    self = FIFOQueue()
    dtypes = vcat(dtypes)
    self.op = get_op(Ops.fifo_queue_v2(capacity=capacity, component_types=dtypes, shapes=shapes, name=name))
    self.dtypes = dtypes
    self
end

@op function RandomShuffleQueue(capacity, dtypes; name=nothing, shapes=nothing)
    self = RandomShuffleQueue()
    if !isa(dtypes, AbstractVector)
        dtypes = [dtypes]
    end
    self.op = get_op(Ops.random_shuffle_queue_v2(capacity=capacity, component_types=dtypes, shapes=shapes, name=name))
    self.dtypes = dtypes
    self
end

@op function enqueue(queue::AbstractQueue, values; name=nothing)
    values = vcat(values)
    Ops.queue_enqueue_v2(Tensor(queue.op), values, Tcomponents=queue.dtypes, name=name)
end

@op function enqueue_many(queue::AbstractQueue, values; name=nothing)
    Ops.queue_enqueue_many_v2(Tensor(queue.op), values, name=name)
end

@op function dequeue(queue::AbstractQueue; name=nothing)
    op = get_op(Ops.dequeue(Tensor(queue.op), component_types=queue.dtypes), name=name)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

@op function dequeue_many(queue::AbstractQueue, n; name=nothing)
    op = get_op(Ops.dequeue_many(Tensor(queue.op), n, component_types=queue.dtypes, name=name))
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

@op function Base.size(queue::AbstractQueue; name=nothing)
    Ops.queue_size_v2(Tensor(queue.op), name=name)
end

@op function Base.close(queue::AbstractQueue; name=nothing)
    Ops.queue_close_v2(Tensor(queue.op), name=name)
end
