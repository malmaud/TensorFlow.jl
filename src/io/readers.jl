using Compat

abstract type AbstractReader end

macro reader(name)
    quote
        mutable struct $(esc(name)) <: AbstractReader
            op::tf.Operation
        end
    end
end

@reader WholeFileReader
@reader TextLineReader
@reader IdentityReader
@reader TFRecordReader
@reader FixedLengthRecordReader

@op function WholeFileReader(; kwargs...)
    WholeFileReader(get_op(Ops.whole_file_reader_v2(; kwargs...)))
end

@op function TextLineReader(; kwargs...)
    TextLineReader(get_op(Ops.text_line_reader_v2(; kwargs...)))
end

@op function Base.read(reader::AbstractReader, queue::tf.AbstractQueue; kwargs...)
    Ops.reader_read_v2(Tensor(reader.op), Tensor(queue.op); kwargs...)
end
