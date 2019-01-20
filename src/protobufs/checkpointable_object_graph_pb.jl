# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct CheckpointableObjectGraph_CheckpointableObject_ObjectReference <: ProtoType
    node_id::Int32
    local_name::AbstractString
    CheckpointableObjectGraph_CheckpointableObject_ObjectReference(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CheckpointableObjectGraph_CheckpointableObject_ObjectReference

mutable struct CheckpointableObjectGraph_CheckpointableObject_SerializedTensor <: ProtoType
    name::AbstractString
    full_name::AbstractString
    checkpoint_key::AbstractString
    optional_restore::Bool
    CheckpointableObjectGraph_CheckpointableObject_SerializedTensor(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CheckpointableObjectGraph_CheckpointableObject_SerializedTensor

mutable struct CheckpointableObjectGraph_CheckpointableObject_SlotVariableReference <: ProtoType
    original_variable_node_id::Int32
    slot_name::AbstractString
    slot_variable_node_id::Int32
    CheckpointableObjectGraph_CheckpointableObject_SlotVariableReference(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CheckpointableObjectGraph_CheckpointableObject_SlotVariableReference

mutable struct CheckpointableObjectGraph_CheckpointableObject <: ProtoType
    children::Base.Vector{CheckpointableObjectGraph_CheckpointableObject_ObjectReference}
    attributes::Base.Vector{CheckpointableObjectGraph_CheckpointableObject_SerializedTensor}
    slot_variables::Base.Vector{CheckpointableObjectGraph_CheckpointableObject_SlotVariableReference}
    CheckpointableObjectGraph_CheckpointableObject(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CheckpointableObjectGraph_CheckpointableObject

mutable struct CheckpointableObjectGraph <: ProtoType
    nodes::Base.Vector{CheckpointableObjectGraph_CheckpointableObject}
    CheckpointableObjectGraph(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CheckpointableObjectGraph

export CheckpointableObjectGraph_CheckpointableObject_ObjectReference, CheckpointableObjectGraph_CheckpointableObject_SerializedTensor, CheckpointableObjectGraph_CheckpointableObject_SlotVariableReference, CheckpointableObjectGraph_CheckpointableObject, CheckpointableObjectGraph
