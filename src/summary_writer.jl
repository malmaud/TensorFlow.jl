type SummaryWriter
    log_dir::String
    SummaryWriter() = new()
end

function SummaryWriter(log_dir)
    self = SummaryWriter()
    self.log_dir = log_dir
    proc = py_proc[]
    remotecall_wait(proc, joinpath(log_dir, "events")) do path
        open_events_file(path)
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

function Base.close(writer::SummaryWriter)
    remotecall_wait(py_proc[]) do
        close_events_file()
    end
    nothing
end
