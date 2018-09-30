# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

struct __enum_ApiDef_Visibility <: ProtoEnum
    DEFAULT_VISIBILITY::Int32
    VISIBLE::Int32
    SKIP::Int32
    HIDDEN::Int32
    __enum_ApiDef_Visibility() = new(0,1,2,3)
end #struct __enum_ApiDef_Visibility
const ApiDef_Visibility = __enum_ApiDef_Visibility()

mutable struct ApiDef_Endpoint <: ProtoType
    name::AbstractString
    deprecated::Bool
    ApiDef_Endpoint(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ApiDef_Endpoint
const __fnum_ApiDef_Endpoint = Int[1,3]
meta(t::Type{ApiDef_Endpoint}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ApiDef_Endpoint, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ApiDef_Arg <: ProtoType
    name::AbstractString
    rename_to::AbstractString
    description::AbstractString
    ApiDef_Arg(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ApiDef_Arg

mutable struct ApiDef_Attr <: ProtoType
    name::AbstractString
    rename_to::AbstractString
    default_value::AttrValue
    description::AbstractString
    ApiDef_Attr(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ApiDef_Attr

mutable struct ApiDef <: ProtoType
    graph_op_name::AbstractString
    deprecation_message::AbstractString
    visibility::Int32
    endpoint::Base.Vector{ApiDef_Endpoint}
    in_arg::Base.Vector{ApiDef_Arg}
    out_arg::Base.Vector{ApiDef_Arg}
    arg_order::Base.Vector{AbstractString}
    attr::Base.Vector{ApiDef_Attr}
    summary::AbstractString
    description::AbstractString
    description_prefix::AbstractString
    description_suffix::AbstractString
    ApiDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ApiDef
const __fnum_ApiDef = Int[1,12,2,3,4,5,11,6,7,8,9,10]
meta(t::Type{ApiDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ApiDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ApiDefs <: ProtoType
    op::Base.Vector{ApiDef}
    ApiDefs(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ApiDefs

export ApiDef_Visibility, ApiDef_Endpoint, ApiDef_Arg, ApiDef_Attr, ApiDef, ApiDefs
