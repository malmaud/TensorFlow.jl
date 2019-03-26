using ProtoBuf
using CRC32c
import ..TensorFlow
const tf = TensorFlow
import ..TensorFlow: tensorflow, Graph, get_def_graph, @py_proc
export FileWriter

struct FileWriter <: tf.Context
    file_handle
    logdir::String
end

"""
    FileWriter(logdir; graph=get_def_graph())

The `FileWriter` type provides a mechanism to create an event file in a
given directory and add summaries and events to it. The class updates the
file contents asynchronously. This allows a training program to call methods
to add data to the file directly from the training loop, without slowing down
training.

On construction the summary writer creates a new event file in `logdir`.

If you pass a `Graph` to the constructor it is added to
the event file.

Arguments:

* logdir: A string. Directory where event file will be written.
* graph: A `Graph` object.
"""
function FileWriter(log_dir::AbstractString; graph=nothing)
    if !tf.in_eager_mode() && graph === nothing
        graph = get_def_graph()
    end
    mkpath(log_dir)
    local path
    for i in Iterators.countfrom(1)
        path = joinpath(log_dir, "events.out.tfevents.$i")
        ispath(path) || break
    end
    writer = FileWriter(open(path, "w"), String(log_dir))
    if graph !== nothing
        write(writer, graph)
    end
    return writer
end

function masked_crc(data)
    x = CRC32c.crc32c(data)
    ((x>>15) | (x<<17)) + 0xa282ead8
end

function Base.write(writer::FileWriter, event::tensorflow.Event)
    b = IOBuffer()
    writeproto(b, event)
    seekstart(b)
    proto = read(b)
    file = writer.file_handle
    proto_length = UInt64(length(proto))
    buffer = IOBuffer()
    write(buffer, proto_length)
    seekstart(buffer)
    proto_length_bytes = read(buffer)
    write(file, proto_length_bytes)
    write(file, masked_crc(proto_length_bytes))
    write(file, proto)
    write(file, masked_crc(proto))
    flush(file)
    nothing
end

function Base.write(writer::FileWriter, summary::tensorflow.Summary, global_step=0)
    event = tensorflow.Event()
    setproperty!(event, :step, Int(global_step))
    setproperty!(event, :wall_time, time())
    setproperty!(event, :summary, summary)
    # Some bug in ProtoBuf.jl is causing these to not be marked as filled,
    # so we do it manually. 
    fillset(event, :wall_time)
    fillset(event, :step)
    fillset(event, :summary)
    
    write(writer, event)
end

function Base.write(writer::FileWriter, bytes::String, global_step=0)
    b = IOBuffer()
    s = tensorflow.Summary()
    write(b, Vector{UInt8}(bytes))
    seekstart(b)
    readproto(b, s)
    write(writer, s, global_step)
end

function Base.write(writer::FileWriter, graph::Graph)
    event = tensorflow.Event()
    setproperty!(event, :graph_def, tf.get_proto(graph))
    write(writer, event)
end

function set_default(writer::FileWriter)
    push!(tf.global_context, writer)
end

function with_default(writer::FileWriter, block)
    tf.with_context(block, writer)
end

function get_default_file_writer()
    return tf.context_value(FileWriter)
end

function record_summary(summary_pb; step=0)
    writer = get_default_file_writer()
    writer === nothing && return
    write(writer, summary_pb, step)
end

function Base.close(writer::FileWriter)
    close(writer.file_handle)
    nothing
end
