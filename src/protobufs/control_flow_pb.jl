# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct ValuesDef_ExternalValuesEntry <: ProtoType
    key::AbstractString
    value::AbstractString
    ValuesDef_ExternalValuesEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ValuesDef_ExternalValuesEntry (mapentry)

mutable struct ValuesDef <: ProtoType
    values::Base.Vector{AbstractString}
    external_values::Base.Dict{AbstractString,AbstractString} # map entry
    ValuesDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ValuesDef

mutable struct ControlFlowContextDef <: ProtoType
    cond_ctxt::Base.Any
    while_ctxt::Base.Any
    ControlFlowContextDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ControlFlowContextDef (has cyclic type dependency)
const __ftype_ControlFlowContextDef = Dict(:cond_ctxt => "CondContextDef", :while_ctxt => "WhileContextDef")
const __oneofs_ControlFlowContextDef = Int[1,1]
const __oneof_names_ControlFlowContextDef = [Symbol("ctxt")]
meta(t::Type{ControlFlowContextDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_ControlFlowContextDef, __oneof_names_ControlFlowContextDef, __ftype_ControlFlowContextDef)

mutable struct CondContextDef <: ProtoType
    context_name::AbstractString
    pred_name::AbstractString
    pivot_name::AbstractString
    branch::Int32
    values_def::ValuesDef
    nested_contexts::Base.Vector{ControlFlowContextDef}
    CondContextDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CondContextDef (has cyclic type dependency)

mutable struct WhileContextDef <: ProtoType
    context_name::AbstractString
    parallel_iterations::Int32
    back_prop::Bool
    swap_memory::Bool
    pivot_name::AbstractString
    pivot_for_pred_name::AbstractString
    pivot_for_body_name::AbstractString
    loop_exit_names::Base.Vector{AbstractString}
    loop_enter_names::Base.Vector{AbstractString}
    values_def::ValuesDef
    maximum_iterations_name::AbstractString
    nested_contexts::Base.Vector{ControlFlowContextDef}
    WhileContextDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct WhileContextDef (has cyclic type dependency)
const __fnum_WhileContextDef = Int[1,2,3,4,5,6,7,8,10,9,11,12]
meta(t::Type{WhileContextDef}) = meta(t, ProtoBuf.DEF_REQ, __fnum_WhileContextDef, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export ValuesDef_ExternalValuesEntry, ValuesDef, ControlFlowContextDef, CondContextDef, WhileContextDef, ControlFlowContextDef, CondContextDef, WhileContextDef
# mapentries: "ValuesDef_ExternalValuesEntry" => ("AbstractString", "AbstractString")
