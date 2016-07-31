# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type TensorShapeProto_Dim
    size::Int64
    name::AbstractString
    TensorShapeProto_Dim(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TensorShapeProto_Dim
hash(v::TensorShapeProto_Dim) = ProtoBuf.protohash(v)
isequal(v1::TensorShapeProto_Dim, v2::TensorShapeProto_Dim) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorShapeProto_Dim, v2::TensorShapeProto_Dim) = ProtoBuf.protoeq(v1, v2)

type TensorShapeProto
    dim::Array{TensorShapeProto_Dim,1}
    unknown_rank::Bool
    TensorShapeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TensorShapeProto
const __fnum_TensorShapeProto = Int[2,3]
meta(t::Type{TensorShapeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorShapeProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::TensorShapeProto) = ProtoBuf.protohash(v)
isequal(v1::TensorShapeProto, v2::TensorShapeProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorShapeProto, v2::TensorShapeProto) = ProtoBuf.protoeq(v1, v2)

export TensorShapeProto_Dim, TensorShapeProto
