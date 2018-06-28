# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct OpDeprecation
    version::Int32
    explanation::AbstractString
    OpDeprecation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type OpDeprecation
hash(v::OpDeprecation) = ProtoBuf.protohash(v)
isequal(v1::OpDeprecation, v2::OpDeprecation) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpDeprecation, v2::OpDeprecation) = ProtoBuf.protoeq(v1, v2)

mutable struct OpDef_ArgDef
    name::AbstractString
    description::AbstractString
    _type::Int32
    type_attr::AbstractString
    number_attr::AbstractString
    type_list_attr::AbstractString
    is_ref::Bool
    OpDef_ArgDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type OpDef_ArgDef
const __fnum_OpDef_ArgDef = Int[1,2,3,4,5,6,16]
meta(t::Type{OpDef_ArgDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OpDef_ArgDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::OpDef_ArgDef) = ProtoBuf.protohash(v)
isequal(v1::OpDef_ArgDef, v2::OpDef_ArgDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpDef_ArgDef, v2::OpDef_ArgDef) = ProtoBuf.protoeq(v1, v2)

mutable struct OpDef_AttrDef
    name::AbstractString
    _type::AbstractString
    default_value::AttrValue
    description::AbstractString
    has_minimum::Bool
    minimum::Int64
    allowed_values::AttrValue
    OpDef_AttrDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type OpDef_AttrDef
hash(v::OpDef_AttrDef) = ProtoBuf.protohash(v)
isequal(v1::OpDef_AttrDef, v2::OpDef_AttrDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpDef_AttrDef, v2::OpDef_AttrDef) = ProtoBuf.protoeq(v1, v2)

mutable struct OpDef
    name::AbstractString
    input_arg::Array{OpDef_ArgDef,1}
    output_arg::Array{OpDef_ArgDef,1}
    attr::Array{OpDef_AttrDef,1}
    deprecation::OpDeprecation
    summary::AbstractString
    description::AbstractString
    is_commutative::Bool
    is_aggregate::Bool
    is_stateful::Bool
    allows_uninitialized_input::Bool
    OpDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type OpDef
const __fnum_OpDef = Int[1,2,3,4,8,5,6,18,16,17,19]
meta(t::Type{OpDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OpDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::OpDef) = ProtoBuf.protohash(v)
isequal(v1::OpDef, v2::OpDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpDef, v2::OpDef) = ProtoBuf.protoeq(v1, v2)

mutable struct OpList
    op::Array{OpDef,1}
    OpList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type OpList
hash(v::OpList) = ProtoBuf.protohash(v)
isequal(v1::OpList, v2::OpList) = ProtoBuf.protoisequal(v1, v2)
==(v1::OpList, v2::OpList) = ProtoBuf.protoeq(v1, v2)

export OpDef_ArgDef, OpDef_AttrDef, OpDef, OpDeprecation, OpList
