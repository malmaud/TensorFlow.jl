# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct TensorDescription
    dtype::Int32
    shape::TensorShapeProto
    allocation_description::AllocationDescription
    TensorDescription(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TensorDescription
const __fnum_TensorDescription = Int[1,2,4]
meta(t::Type{TensorDescription}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorDescription, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::TensorDescription) = ProtoBuf.protohash(v)
isequal(v1::TensorDescription, v2::TensorDescription) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorDescription, v2::TensorDescription) = ProtoBuf.protoeq(v1, v2)

export TensorDescription
