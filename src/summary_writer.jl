using ProtoBuf
import TensorFlow
const tf = TensorFlow
import ..TensorFlow: tensorflow, Graph, get_def_graph, @py_proc

immutable FileWriter
    pyo::Future
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
function FileWriter(log_dir::AbstractString; graph=get_def_graph())
    path = joinpath(log_dir, "events")
    pyo = @py_proc pywrap_tensorflow[][:EventsWriter](py_bytes($path))
    writer = FileWriter(pyo, String(log_dir))
    if graph !== nothing
        write(writer, graph)
    end
    return writer
end

function Base.write(writer::FileWriter, event::tensorflow.Event)
    b = IOBuffer()
    writeproto(b, event)
    seekstart(b)
    proto = read(b)
    @py_proc begin
        py_event = py_tf[][:Event]()
        py_event[:ParseFromString](py_bytes($(proto)))
        $(writer.pyo)[:WriteEvent](py_event)
        $(writer.pyo)[:Flush]()
    end
    nothing
end

function Base.write(writer::FileWriter, summary::tensorflow.Summary, global_step=0)
    event = tensorflow.Event()
    set_field!(event, :step, Int(global_step))
    set_field!(event, :wall_time, time())
    set_field!(event, :summary, summary)
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
    set_field!(event, :graph_def, tf.get_proto(graph))
    write(writer, event)
end

function Base.close(writer::FileWriter)
    @py_proc $(writer.pyo)[:Close]()
    nothing
end
