# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct TensorDescription <: ProtoType
    dtype::Int32
    shape::TensorShapeProto
    allocation_description::AllocationDescription
    TensorDescription(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorDescription
const __fnum_TensorDescription = Int[1,2,4]
meta(t::Type{TensorDescription}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorDescription, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export TensorDescription
