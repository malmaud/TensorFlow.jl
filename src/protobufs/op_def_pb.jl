# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct OpDeprecation <: ProtoType
    version::Int32
    explanation::AbstractString
    OpDeprecation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpDeprecation

mutable struct OpDef_ArgDef <: ProtoType
    name::AbstractString
    description::AbstractString
    _type::Int32
    type_attr::AbstractString
    number_attr::AbstractString
    type_list_attr::AbstractString
    is_ref::Bool
    OpDef_ArgDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpDef_ArgDef
const __fnum_OpDef_ArgDef = Int[1,2,3,4,5,6,16]
meta(t::Type{OpDef_ArgDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OpDef_ArgDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct OpDef_AttrDef <: ProtoType
    name::AbstractString
    _type::AbstractString
    default_value::AttrValue
    description::AbstractString
    has_minimum::Bool
    minimum::Int64
    allowed_values::AttrValue
    OpDef_AttrDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpDef_AttrDef

mutable struct OpDef <: ProtoType
    name::AbstractString
    input_arg::Base.Vector{OpDef_ArgDef}
    output_arg::Base.Vector{OpDef_ArgDef}
    attr::Base.Vector{OpDef_AttrDef}
    deprecation::OpDeprecation
    summary::AbstractString
    description::AbstractString
    is_commutative::Bool
    is_aggregate::Bool
    is_stateful::Bool
    allows_uninitialized_input::Bool
    OpDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpDef
const __fnum_OpDef = Int[1,2,3,4,8,5,6,18,16,17,19]
meta(t::Type{OpDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OpDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct OpList <: ProtoType
    op::Base.Vector{OpDef}
    OpList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OpList

export OpDef_ArgDef, OpDef_AttrDef, OpDef, OpDeprecation, OpList
