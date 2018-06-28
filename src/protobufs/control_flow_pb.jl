# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct ValuesDef_ExternalValuesEntry
    key::AbstractString
    value::AbstractString
    ValuesDef_ExternalValuesEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ValuesDef_ExternalValuesEntry (mapentry)
hash(v::ValuesDef_ExternalValuesEntry) = ProtoBuf.protohash(v)
isequal(v1::ValuesDef_ExternalValuesEntry, v2::ValuesDef_ExternalValuesEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::ValuesDef_ExternalValuesEntry, v2::ValuesDef_ExternalValuesEntry) = ProtoBuf.protoeq(v1, v2)

mutable struct ValuesDef
    values::Array{AbstractString,1}
    external_values::Dict{AbstractString,AbstractString} # map entry
    ValuesDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ValuesDef
hash(v::ValuesDef) = ProtoBuf.protohash(v)
isequal(v1::ValuesDef, v2::ValuesDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::ValuesDef, v2::ValuesDef) = ProtoBuf.protoeq(v1, v2)

mutable struct CondContextDef
    context_name::AbstractString
    pred_name::AbstractString
    pivot_name::AbstractString
    branch::Int32
    values_def::ValuesDef
    CondContextDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CondContextDef
hash(v::CondContextDef) = ProtoBuf.protohash(v)
isequal(v1::CondContextDef, v2::CondContextDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::CondContextDef, v2::CondContextDef) = ProtoBuf.protoeq(v1, v2)

mutable struct WhileContextDef
    context_name::AbstractString
    parallel_iterations::Int32
    back_prop::Bool
    swap_memory::Bool
    pivot_name::AbstractString
    pivot_for_pred_name::AbstractString
    pivot_for_body_name::AbstractString
    loop_exit_names::Array{AbstractString,1}
    values_def::ValuesDef
    WhileContextDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type WhileContextDef
hash(v::WhileContextDef) = ProtoBuf.protohash(v)
isequal(v1::WhileContextDef, v2::WhileContextDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::WhileContextDef, v2::WhileContextDef) = ProtoBuf.protoeq(v1, v2)

export ValuesDef_ExternalValuesEntry, ValuesDef, CondContextDef, WhileContextDef
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("ValuesDef_ExternalValuesEntry",("AbstractString","AbstractString"))
