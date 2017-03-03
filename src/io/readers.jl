using Compat

@compat abstract type AbstractReader end

macro reader(name)
    quote
        type $(esc(name)) <: AbstractReader
            op::tf.Operation
        end
    end
end

@reader WholeFileReader
@reader TextLineReader
@reader IdentityReader
@reader TFRecordReader
@reader FixedLengthRecordReader

function WholeFileReader(; name="WholeFileReader")
    local desc
    with_op_name(name) do
        desc = NodeDescription("WholeFileReader")
    end
    return WholeFileReader(Operation(desc))
end

function TextLineReader(skip_header_lines::Int=0; name="TextLineReader")
    local desc
    with_op_name(name) do
        desc = NodeDescription("TextLineReader")
        desc["skip_header_lines"] = Int64(skip_header_lines)
    end
    return TextLineReader(Operation(desc))
end

function Base.read(reader::AbstractReader, queue::tf.AbstractQueue; name=nothing)
    local desc
    with_op_name(name, "ReaderRead") do
        desc = NodeDescription("ReaderRead")
        add_input(desc, Tensor(reader.op))
        add_input(desc, Tensor(queue.op))
    end
    op = tf.Operation(desc)
    tf.Tensor(op, 1), tf.Tensor(op, 2)
end
