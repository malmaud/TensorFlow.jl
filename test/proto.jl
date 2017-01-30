# Test loading/saving of protos

using TensorFlow: load_proto

let
    t = get_def(constant("test")).attr["value"].tensor
    @test load_proto(t) == Vector{UInt8}("test")
end

let
    val = ["a" "bc"; "de" "fghi"]
    t = get_def(constant(val)).attr["value"].tensor
    @test all(load_proto(t) .== map(Vector{UInt8}, val))
end
    
