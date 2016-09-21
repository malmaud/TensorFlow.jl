using ProtoBuf
import ..TensorFlow: tensorflow, Graph

type SummaryWriter
    log_dir::String
    SummaryWriter() = new()
end

function SummaryWriter(log_dir; graph=nothing)
    self = SummaryWriter()
    self.log_dir = log_dir
    path = joinpath(log_dir, "events")
    tf.open_events_file(path)
    if graph !== nothing
        write(self, graph)
    end
    self
end

function Base.write(writer::SummaryWriter, event::tensorflow.Event)
    b = IOBuffer()
    writeproto(b, event)
    seekstart(b)
    proto = read(b)
    tf.write_event(proto)
    nothing
end

function Base.write(writer::SummaryWriter, summary::tensorflow.Summary, global_step=0)
    event = tensorflow.Event()
    set_field!(event, :step, Int(global_step))
    set_field!(event, :wall_time, time())
    set_field!(event, :summary, summary)
    write(writer, event)
end

function Base.write(writer::SummaryWriter, bytes::String, global_step=0)
    b = IOBuffer()
    s = tensorflow.Summary()
    write(b, bytes.data)
    seekstart(b)
    readproto(b, s)
    write(writer, s, global_step)
end

function Base.write(writer::SummaryWriter, graph::Graph)
    event = tensorflow.Event()
    set_field!(event, :graph_def, tf.get_proto(graph))
    write(writer, event)
end

function Base.close(writer::SummaryWriter)
    tf.close_events_file()
    nothing
end
