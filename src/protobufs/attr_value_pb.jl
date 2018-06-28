# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct AttrValue_ListValue
    s::Array{Array{UInt8,1},1}
    i::Array{Int64,1}
    f::Array{Float32,1}
    b::Array{Bool,1}
    _type::Array{Int32,1}
    shape::Array{TensorShapeProto,1}
    tensor::Array{TensorProto,1}
    AttrValue_ListValue(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type AttrValue_ListValue
const __fnum_AttrValue_ListValue = Int[2,3,4,5,6,7,8]
const __pack_AttrValue_ListValue = Symbol[:i,:f,:b,:_type]
meta(t::Type{AttrValue_ListValue}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AttrValue_ListValue, ProtoBuf.DEF_VAL, true, __pack_AttrValue_ListValue, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::AttrValue_ListValue) = ProtoBuf.protohash(v)
isequal(v1::AttrValue_ListValue, v2::AttrValue_ListValue) = ProtoBuf.protoisequal(v1, v2)
==(v1::AttrValue_ListValue, v2::AttrValue_ListValue) = ProtoBuf.protoeq(v1, v2)

mutable struct AttrValue
    s::Array{UInt8,1}
    i::Int64
    f::Float32
    b::Bool
    _type::Int32
    shape::TensorShapeProto
    tensor::TensorProto
    list::AttrValue_ListValue
    placeholder::AbstractString
    AttrValue(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type AttrValue
const __fnum_AttrValue = Int[2,3,4,5,6,7,8,1,9]
const __oneofs_AttrValue = Int[1,1,1,1,1,1,1,1,1]
const __oneof_names_AttrValue = [Symbol("value")]
meta(t::Type{AttrValue}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AttrValue, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_AttrValue, __oneof_names_AttrValue)
hash(v::AttrValue) = ProtoBuf.protohash(v)
isequal(v1::AttrValue, v2::AttrValue) = ProtoBuf.protoisequal(v1, v2)
==(v1::AttrValue, v2::AttrValue) = ProtoBuf.protoeq(v1, v2)

mutable struct NameAttrList_AttrEntry
    key::AbstractString
    value::AttrValue
    NameAttrList_AttrEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NameAttrList_AttrEntry (mapentry)
hash(v::NameAttrList_AttrEntry) = ProtoBuf.protohash(v)
isequal(v1::NameAttrList_AttrEntry, v2::NameAttrList_AttrEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::NameAttrList_AttrEntry, v2::NameAttrList_AttrEntry) = ProtoBuf.protoeq(v1, v2)

mutable struct NameAttrList
    name::AbstractString
    attr::Dict{AbstractString,AttrValue} # map entry
    NameAttrList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NameAttrList
hash(v::NameAttrList) = ProtoBuf.protohash(v)
isequal(v1::NameAttrList, v2::NameAttrList) = ProtoBuf.protoisequal(v1, v2)
==(v1::NameAttrList, v2::NameAttrList) = ProtoBuf.protoeq(v1, v2)

export AttrValue_ListValue, AttrValue, NameAttrList_AttrEntry, NameAttrList
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("NameAttrList_AttrEntry",("AbstractString","AttrValue"))
