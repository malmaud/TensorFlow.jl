@compat abstract type AbstractQueue end

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

@op function FIFOQueue(capacity, dtypes; name=nothing, shapes=nothing)
    self = FIFOQueue()
    dtypes = to_list(dtypes)
    local desc
    with_op_name(name, "FIFOQueue") do
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

@op function RandomShuffleQueue(capacity, dtypes; name=nothing, shapes=nothing)
    self = RandomShuffleQueue()
    if !isa(dtypes, AbstractVector)
        dtypes = [dtypes]
    end
    local desc
    with_op_name(name, "RandomShuffleQueue") do
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

@op function enqueue(queue::AbstractQueue, values; name=nothing)
    values = to_list(values)
    local desc
    with_op_name(name, "QueueEnqueue") do
        desc = NodeDescription("QueueEnqueue")
        add_input(desc, queue.op)
        add_input(desc, map(to_tensor, values))
        set_attr_list(desc, "Tcomponents", queue.dtypes)
    end
    Tensor(Operation(desc))
end

@op function enqueue_many(queue::AbstractQueue, values; name=nothing)
    local desc
    with_op_name(name, "QueueEnqueueMany") do
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

@op function dequeue(queue::AbstractQueue; name=nothing)
    local desc
    with_op_name(name, "QueueDequeue") do
        desc = NodeDescription("QueueDequeue")
        add_input(desc, queue.op)
        set_attr_list(desc, "component_types", queue.dtypes)
    end
    op = Operation(desc)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

@op function dequeue_many(queue::AbstractQueue, n; name=nothing)
    local desc
    with_op_name(name, "QueueDequeueMany") do
        desc = NodeDescription("QueueDequeueMany")
        add_input(desc, queue.op)
        add_input(desc, Tensor(Int32(n)))
        set_attr_list(desc, "component_types", queue.dtypes)
    end
    op = Operation(desc)
    [Tensor(op, i) for i in 1:length(queue.dtypes)]
end

@op function Base.size(queue::AbstractQueue; name=nothing)
    local desc
    with_op_name(name, "QueueSize") do
        desc = NodeDescription("QueueSize")
        add_input(desc, queue.op)
    end
    Tensor(Operation(desc))
end

@op function Base.close(queue::AbstractQueue; name=nothing)
    local desc
    with_op_name(name, "QueueClose") do
        desc = NodeDescription("QueueClose")
        add_input(desc, queue.op)
    end
    Tensor(Operation(desc))
end
