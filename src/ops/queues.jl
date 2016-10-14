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

function FIFOQueue(capacity, dtypes; name="FIFOQueue", shapes=nothing)
    self = FIFOQueue()
    dtypes = to_list(dtypes)
    local desc
    with_op_name(name) do
        desc = NodeDescription("FIFOQueue")
        desc["capacity"] = Int64(capacity)
        set_attr_list(desc, "component_types", dtypes)
        if shapes !== nothing
            set_attr_shape_list(desc, "shapes", shapes)
        end
    end
    self.op = Operation(desc)
    self.dtypes = dtypes
    self
end

function RandomShuffleQueue(capacity, dtypes; name="RandomShuffleQueue", shapes=nothing)
    self = RandomShuffleQueue()
    if !isa(dtypes, AbstractVector)
        dtypes = [dtypes]
    end
    local desc
    with_op_name(name) do
        desc = NodeDescription("RandomShuffleQueue")
        desc["capacity"] = Int64(capacity)
        set_attr_list(desc, "component_types", dtypes)
        if shapes !== nothing
            set_attr_shape_list(desc, "shapes", shapes)
        end
    end
    self.op = Operation(desc)
    self.dtypes = dtypes
    self
end

function enqueue(queue::AbstractQueue, values; name="QueueEnqueue")
    values = to_list(values)
    local desc
    with_op_name(name) do
        desc = NodeDescription("QueueEnqueue")
        add_input(desc, queue.op)
        add_input(desc, map(to_tensor, values))
        set_attr_list(desc, "Tcomponents", queue.dtypes)
    end
    Tensor(Operation(desc))
end

function enqueue_many(queue::AbstractQueue, values; name="QueueEnqueueMany")
    local desc
    with_op_name(name) do
        desc = NodeDescription("QueueEnqueueMany")
        add_input(desc, queue.op)
        if isa(values, AbstractVector) || isa(values, Tuple)
            add_input(desc, [values...])
        else
            add_input(desc, [values])
        end
    end
    Tensor(Operation(desc))
end

function dequeue(queue::AbstractQueue; name="QueueDequeue")
    local desc
    with_op_name(name) do
        desc = NodeDescription("QueueDequeue")
        add_input(desc, queue.op)
        set_attr_list(desc, "component_types", queue.dtypes)
    end
    op = Operation(desc)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

function dequeue_many(queue::AbstractQueue, n; name="QueueDequeueMany")
    local desc
    with_op_name(name) do
        desc = NodeDescription("QueueDequeueMany")
        add_input(desc, queue.op)
        add_input(desc, Tensor(Int32(n)))
        set_attr_list(desc, "component_types", queue.dtypes)
    end
    op = Operation(desc)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

function Base.size(queue::AbstractQueue; name="QueueSize")
    local desc
    with_op_name(name) do
        desc = NodeDescription("QueueSize")
        add_input(desc, queue.op)
    end
    Tensor(Operation(desc))
end

function Base.close(queue::AbstractQueue; name="QueueClose")
    local desc
    with_op_name(name) do
        desc = NodeDescription("QueueClose")
        add_input(desc, queue.op)
    end
    Tensor(Operation(desc))
end
