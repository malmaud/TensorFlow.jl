module TFRecord

export
RecordWriter,
RecordIterator

import TensorFlow
const tf = TensorFlow
using PyCall

immutable RecordWriter
    pyo::Future
end

"""
    `RecordWriter(path)`

Opens a TensorFlow record writer.

Records will be written to the file at the given path.
"""
function RecordWriter(path::AbstractString)
    pyo = @tf.py_proc py_tf[][:python_io][:TFRecordWriter]($path)
    RecordWriter(pyo)
end

"""
    `write(writer::RecordWriter, msg)`

Writes a record `msg` to the TensorFlow writer `writer`. Tries to convert
the msg to `Vector{UInt8}` before writing.
"""
function Base.write(writer::RecordWriter, msg::Vector{UInt8})
    fetch(@tf.py_proc $(writer.pyo)[:write](py_bytes($msg)))
end

Base.write(writer::RecordWriter, s::AbstractString) = write(writer, Vector{UInt8}(s))

function RecordWriter(f::Function, path)
    writer = RecordWriter(path)
    f(writer)
    close(writer)
end

function Base.close(writer::RecordWriter)
    fetch(@tf.py_proc $(writer.pyo)[:close]())
end

immutable RecordIterator
    pyo::Future
end

immutable RecordIteratorState
    val::Nullable{Vector{UInt8}}
end

"""
    `RecordIterator(path)`

Returns a Julia iterator that returns the records in the TF Record file
at `path` as `Vector{UInt8}` objects.
"""
function RecordIterator(path::AbstractString)
    pyo = @tf.py_proc py_tf[][:python_io][:tf_record_iterator]($path)
    RecordIterator(pyo)
end

function _next(iter::RecordIterator)
    try
        record = fetch(@tf.py_proc $(iter.pyo)[:__next__]())
        RecordIteratorState(Nullable(record))
    catch err
        RecordIteratorState(Nullable())
    end
end

function Base.start(iter::RecordIterator)
    return _next(iter)
end

function Base.next(iter::RecordIterator, state)
    val = get(state.val)
    return val, _next(iter)
end

function Base.done(iter::RecordIterator, state)
    isnull(state.val)
end

Base.iteratorsize(::Type{RecordIterator}) = Base.SizeUnknown()
Base.eltype(::Type{RecordIterator}) = Vector{UInt8}

end
