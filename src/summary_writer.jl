type SummaryWriter
    log_dir::String
    SummaryWriter() = new()
end

function SummaryWriter(log_dir; graph=nothing)
    self = SummaryWriter()
    self.log_dir = log_dir
    proc = py_proc[]
    remotecall_wait(proc, joinpath(log_dir, "events")) do path
        open_events_file(path)
    end
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
    remotecall_wait(py_proc[], proto) do proto
        write_event(proto)
    end
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
    set_field!(event, :graph_def, get_proto(graph))
    write(writer, event)
end

function Base.close(writer::SummaryWriter)
    remotecall_wait(py_proc[]) do
        close_events_file()
    end
    nothing
end
