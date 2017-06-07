using TensorFlow
using Base.Test

mktemp() do filename, fileio
    TensorFlow.io.RecordWriter(filename) do writer
        write(writer, UInt8[1,2])
        write(writer, "hi")
    end

    reader = TensorFlow.io.RecordIterator(filename)
    records = collect(reader)
    @test records[1] == UInt8[1, 2]
    @test records[2] == Vector{UInt8}("hi")
end
