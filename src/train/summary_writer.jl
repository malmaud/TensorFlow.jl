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
    eval(Main, quote
        remotecall_wait($(tf.pyproc[])) do
            open_events_file($path)
        end
    end)
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
    eval(Main, quote
        remotecall_wait($(tf.pyproc[])) do
            write_event($proto)
        end
    end)
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
    eval(Main, quote
        remotecall_wait($(tf.pyproc[])) do
            close_events_file()
        end
    end)
    nothing
end
