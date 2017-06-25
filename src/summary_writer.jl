using ProtoBuf
import TensorFlow
const tf = TensorFlow
import ..TensorFlow: tensorflow, Graph, get_def_graph, @py_proc

immutable FileWriter
    pyo::Future
    logdir::String
end

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
