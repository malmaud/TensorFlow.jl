# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type TensorProto
    dtype::Int32
    tensor_shape::TensorShapeProto
    version_number::Int32
    tensor_content::Array{UInt8,1}
    half_val::Array{Int32,1}
    float_val::Array{Float32,1}
    double_val::Array{Float64,1}
    int_val::Array{Int32,1}
    string_val::Array{Array{UInt8,1},1}
    scomplex_val::Array{Float32,1}
    int64_val::Array{Int64,1}
    bool_val::Array{Bool,1}
    dcomplex_val::Array{Float64,1}
    TensorProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TensorProto
const __fnum_TensorProto = Int[1,2,3,4,13,5,6,7,8,9,10,11,12]
const __pack_TensorProto = Symbol[:half_val,:float_val,:double_val,:int_val,:scomplex_val,:int64_val,:bool_val,:dcomplex_val]
meta(t::Type{TensorProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorProto, ProtoBuf.DEF_VAL, true, __pack_TensorProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::TensorProto) = ProtoBuf.protohash(v)
isequal(v1::TensorProto, v2::TensorProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorProto, v2::TensorProto) = ProtoBuf.protoeq(v1, v2)

export TensorProto
