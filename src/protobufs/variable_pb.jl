# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct SaveSliceInfoDef <: ProtoType
    full_name::AbstractString
    full_shape::Base.Vector{Int64}
    var_offset::Base.Vector{Int64}
    var_shape::Base.Vector{Int64}
    SaveSliceInfoDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SaveSliceInfoDef
const __pack_SaveSliceInfoDef = Symbol[:full_shape,:var_offset,:var_shape]
meta(t::Type{SaveSliceInfoDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_SaveSliceInfoDef, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct VariableDef <: ProtoType
    variable_name::AbstractString
    initial_value_name::AbstractString
    initializer_name::AbstractString
    snapshot_name::AbstractString
    save_slice_info_def::SaveSliceInfoDef
    is_resource::Bool
    trainable::Bool
    VariableDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct VariableDef
const __fnum_VariableDef = Int[1,6,2,3,4,5,7]
meta(t::Type{VariableDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_VariableDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export VariableDef, SaveSliceInfoDef
