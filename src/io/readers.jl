abstract AbstractReader

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

function WholeFileReader(; name="")
    desc = tf.NodeDescription("WholeFileReader", tf.get_name(name))
    op = tf.Operation(desc)
    return WholeFileReader(op)
end

function TextLineReader(skip_header_lines::Int=0; name="")
    desc = tf.NodeDescription("TextLineReader", tf.get_name(name))
    desc["skip_header_lines"] = Int64(skip_header_lines)
    op = tf.Operation(desc)
    return TextLineReader(op)
end

function Base.read(reader::AbstractReader, queue::tf.AbstractQueue; name="")
    desc = tf.NodeDescription("ReaderRead", tf.get_name(name))
    tf.add_input(desc, tf.Tensor(reader.op))
    tf.add_input(desc, tf.Tensor(queue.op))
    op = tf.Operation(desc)
    tf.Tensor(op, 1), tf.Tensor(op, 2)
end
