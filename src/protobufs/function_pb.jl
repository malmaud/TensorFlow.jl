# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct FunctionDef_AttrEntry <: ProtoType
    key::AbstractString
    value::AttrValue
    FunctionDef_AttrEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct FunctionDef_AttrEntry (mapentry)

mutable struct FunctionDef_RetEntry <: ProtoType
    key::AbstractString
    value::AbstractString
    FunctionDef_RetEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct FunctionDef_RetEntry (mapentry)

mutable struct FunctionDef <: ProtoType
    signature::OpDef
    attr::Base.Dict{AbstractString,AttrValue} # map entry
    node_def::Base.Vector{NodeDef}
    ret::Base.Dict{AbstractString,AbstractString} # map entry
    FunctionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct FunctionDef
const __fnum_FunctionDef = Int[1,5,3,4]
meta(t::Type{FunctionDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_FunctionDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GradientDef <: ProtoType
    function_name::AbstractString
    gradient_func::AbstractString
    GradientDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GradientDef

mutable struct FunctionDefLibrary <: ProtoType
    _function::Base.Vector{FunctionDef}
    gradient::Base.Vector{GradientDef}
    FunctionDefLibrary(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct FunctionDefLibrary

export FunctionDefLibrary, FunctionDef_AttrEntry, FunctionDef_RetEntry, FunctionDef, GradientDef
# mapentries: "FunctionDef_AttrEntry" => ("AbstractString", "AttrValue"), "FunctionDef_RetEntry" => ("AbstractString", "AbstractString")
