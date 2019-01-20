# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

struct __enum_BundleHeaderProto_Endianness <: ProtoEnum
    LITTLE::Int32
    BIG::Int32
    __enum_BundleHeaderProto_Endianness() = new(0,1)
end #struct __enum_BundleHeaderProto_Endianness
const BundleHeaderProto_Endianness = __enum_BundleHeaderProto_Endianness()

mutable struct BundleHeaderProto <: ProtoType
    num_shards::Int32
    endianness::Int32
    version::VersionDef
    BundleHeaderProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BundleHeaderProto

mutable struct BundleEntryProto <: ProtoType
    dtype::Int32
    shape::TensorShapeProto
    shard_id::Int32
    offset::Int64
    size::Int64
    crc32c::UInt32
    slices::Base.Vector{TensorSliceProto}
    BundleEntryProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct BundleEntryProto
const __wtype_BundleEntryProto = Dict(:crc32c => :fixed32)
meta(t::Type{BundleEntryProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_BundleEntryProto, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export BundleHeaderProto_Endianness, BundleHeaderProto, BundleEntryProto
