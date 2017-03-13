using ProtoBuf
import TensorFlow
const tf = TensorFlow
import ..TensorFlow: tensorflow, Graph, get_def_graph, @py_proc

type FileWriter
    log_dir::String
    FileWriter() = new()
end

function FileWriter(log_dir; graph=get_def_graph())
    self = FileWriter()
    self.log_dir = log_dir
    path = joinpath(log_dir, "events")
    @py_proc open_events_file($path)
    if graph !== nothing
        write(self, graph)
    end
    self
end

function Base.write(writer::FileWriter, event::tensorflow.Event)
    b = IOBuffer()
    writeproto(b, event)
    seekstart(b)
    proto = read(b)
    @py_proc write_event($proto)
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
    write(b, bytes.data)
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
    @py_proc close_events_file()
    nothing
end
