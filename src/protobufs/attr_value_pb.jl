# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct AttrValue_ListValue <: ProtoType
    s::Base.Vector{Array{UInt8,1}}
    i::Base.Vector{Int64}
    f::Base.Vector{Float32}
    b::Base.Vector{Bool}
    _type::Base.Vector{Int32}
    shape::Base.Vector{TensorShapeProto}
    tensor::Base.Vector{TensorProto}
    func::Base.Any
    AttrValue_ListValue(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AttrValue_ListValue (has cyclic type dependency)
const __fnum_AttrValue_ListValue = Int[2,3,4,5,6,7,8,9]
const __pack_AttrValue_ListValue = Symbol[:i,:f,:b,:_type]
const __ftype_AttrValue_ListValue = Dict(:func => "Base.Vector{NameAttrList}")
meta(t::Type{AttrValue_ListValue}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AttrValue_ListValue, ProtoBuf.DEF_VAL, true, __pack_AttrValue_ListValue, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_AttrValue_ListValue)

mutable struct AttrValue <: ProtoType
    s::Array{UInt8,1}
    i::Int64
    f::Float32
    b::Bool
    _type::Int32
    shape::TensorShapeProto
    tensor::TensorProto
    list::AttrValue_ListValue
    func::Base.Any
    placeholder::AbstractString
    AttrValue(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AttrValue (has cyclic type dependency)
const __fnum_AttrValue = Int[2,3,4,5,6,7,8,1,10,9]
const __ftype_AttrValue = Dict(:func => "NameAttrList")
const __oneofs_AttrValue = Int[1,1,1,1,1,1,1,1,1,1]
const __oneof_names_AttrValue = [Symbol("value")]
meta(t::Type{AttrValue}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AttrValue, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_AttrValue, __oneof_names_AttrValue, __ftype_AttrValue)

mutable struct NameAttrList_AttrEntry <: ProtoType
    key::AbstractString
    value::AttrValue
    NameAttrList_AttrEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NameAttrList_AttrEntry (mapentry) (has cyclic type dependency)

mutable struct NameAttrList <: ProtoType
    name::AbstractString
    attr::Base.Dict # map entry
    NameAttrList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NameAttrList (has cyclic type dependency)
const __ftype_NameAttrList = Dict(:attr => "Base.Dict{AbstractString,AttrValue}")
meta(t::Type{NameAttrList}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_NameAttrList)

export AttrValue_ListValue, AttrValue, NameAttrList_AttrEntry, NameAttrList, AttrValue_ListValue, AttrValue, NameAttrList_AttrEntry, NameAttrList
# mapentries: "NameAttrList_AttrEntry" => ("AbstractString", "AttrValue")
