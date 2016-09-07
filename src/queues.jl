abstract QueueBase

type FIFOQueue <: QueueBase
    op::Operation
    dtypes
end

function Base.show(io::IO, f::FIFOQueue)
    print(io, "FIFOQueue")
end

type RandomShuffleQueue <: QueueBase
end

function FIFOQueue(capacity, dtypes; name="")
    desc = NodeDescription("FIFOQueue", get_name(name))
    desc["capacity"] = Int64(capacity)
    set_attr_list(desc, "component_types", dtypes)
    FIFOQueue(Operation(desc), dtypes)
end

function enqueue_many(queue::QueueBase, values; name="")
    desc = NodeDescription("QueueEnqueue", get_name(name))
    add_input(desc, queue.op)
    add_input(desc, values)
    Tensor(Operation(desc))
end

function enqueue(queue::QueueBase, value; name="")
    enqueue_many(queue, [Tensor(value)]; name=name)
end

function dequeue(queue::QueueBase; name="")
    desc = NodeDescription("QueueDequeue", get_name(name))
    add_input(desc, queue.op)
    set_attr_list(desc, "component_types", queue.dtypes)
    Tensor(Operation(desc))
end
