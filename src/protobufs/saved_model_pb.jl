# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct SavedModel <: ProtoType
    saved_model_schema_version::Int64
    meta_graphs::Base.Vector{MetaGraphDef}
    SavedModel(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SavedModel

export SavedModel
