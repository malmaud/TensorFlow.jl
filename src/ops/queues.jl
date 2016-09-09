abstract AbstractQueue

type FIFOQueue <: AbstractQueue
    op::Operation
    dtypes
    FIFOQueue() = new()
end

function Base.show(io::IO, f::FIFOQueue)
    print(io, "FIFOQueue")
end

type RandomShuffleQueue <: AbstractQueue
    op::Operation
    dtypes
    RandomShuffleQueue() = new()
end

function FIFOQueue(capacity, dtypes; name="", shapes=nothing)
    self = FIFOQueue()
    dtypes = to_list(dtypes)
    desc = NodeDescription("FIFOQueue", get_name(name))
    desc["capacity"] = Int64(capacity)
    set_attr_list(desc, "component_types", dtypes)
    if shapes !== nothing
        set_attr_shape_list(desc, "shapes", shapes)
    end
    self.op = Operation(desc)
    self.dtypes = dtypes
    self
end

function RandomShuffleQueue(capacity, dtypes; name="", shapes=nothing)
    self = RandomShuffleQueue()
    if !isa(dtypes, AbstractVector)
        dtypes = [dtypes]
    end
    desc = NodeDescription("RandomShuffleQueue", get_name(name))
    desc["capacity"] = Int64(capacity)
    set_attr_list(desc, "component_types", dtypes)
    if shapes !== nothing
        set_attr_shape_list(desc, "shapes", shapes)
    end
    self.op = Operation(desc)
    self.dtypes = dtypes
    self
end

function enqueue(queue::AbstractQueue, values; name="")
    values = to_list(values)
    desc = NodeDescription("QueueEnqueue", get_name(name))
    add_input(desc, queue.op)
    add_input(desc, map(to_tensor, values))
    set_attr_list(desc, "Tcomponents", queue.dtypes)
    Tensor(Operation(desc))
end

function enqueue_many(queue::AbstractQueue, values; name="")
    desc = NodeDescription("QueueEnqueueMany", get_name(name))
    add_input(desc, queue.op)
    if isa(values, AbstractVector) || isa(values, Tuple)
        add_input(desc, [values...])
    else
        add_input(desc, [values])
    end
    Tensor(Operation(desc))
end

function dequeue(queue::AbstractQueue; name="")
    desc = NodeDescription("QueueDequeue", get_name(name))
    add_input(desc, queue.op)
    set_attr_list(desc, "component_types", queue.dtypes)
    op = Operation(desc)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

function dequeue_many(queue::AbstractQueue, n; name="")
    desc = NodeDescription("QueueDequeueMany", get_name(name))
    add_input(desc, queue.op)
    add_input(desc, Tensor(Int32(n)))
    set_attr_list(desc, "component_types", queue.dtypes)
    op = Operation(desc)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

function Base.size(queue::AbstractQueue; name="")
    desc = NodeDescription("QueueSize", get_name(name))
    add_input(desc, queue.op)
    Tensor(Operation(desc))
end

function Base.close(queue::AbstractQueue; name="")
    desc = NodeDescription("QueueClose", get_name(name))
    add_input(desc, queue.op)
    Tensor(Operation(desc))
end
