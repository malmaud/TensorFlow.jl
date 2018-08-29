module TFRecord

export
RecordWriter,
RecordIterator

import TensorFlow
import Distributed
const tf = TensorFlow

using PyCall

struct RecordWriter
    pyo::Distributed.Future
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

struct RecordIterator
    pyo::Distributed.Future
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
        ans=@static if PyCall.pyversion >= v"3.0.0"
            fetch(@tf.py_proc $(iter.pyo)[:__next__]())
        else
            #Python 2
            fetch(@tf.py_proc $(iter.pyo)[:next]())
        end
        Vector{UInt8}(ans)
    catch err
        if isa(err, Distributed.RemoteException) && isa(err.captured.ex, PyCall.PyError)
            # Only catch it, if it could be an  StopIteration exception thrown in python
            # which signifies the end of iteration being reached normally
            nothing # signal to stop julia iteration
        else
            rethrow(err)
        end
    end
end

function Base.iterate(iter::RecordIterator, state=iter)
	record = _next(iter)
	if record isa Nothing
		nothing # end iteration
	else
		(record, iter)
	end
end

#function Base.start(iter::RecordIterator)
#    return _next(iter)
#end

#function Base.next(iter::RecordIterator, state)
#    val = get(state.val)
#    return val, _next(iter)
#end

#function Base.done(iter::RecordIterator, state)
#    isnull(state.val)
#end

#Base.IteratorSize(::RecordIterator) = Base.SizeUnknown()
#Base.IteratorEltype(::RecordIterator) = Vector{UInt8}

Base.IteratorSize(::RecordIterator) = Base.SizeUnknown()
Base.eltype(::RecordIterator) = Vector{UInt8}

end
