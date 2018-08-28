# Test loading/saving of protos

using TensorFlow: load_proto

let
    t = get_def(constant("test")).attr["value"].tensor
    @test convert(Array, load_proto(t)) == Vector{UInt8}("test")
end
