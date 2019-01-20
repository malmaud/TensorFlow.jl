# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct JobDef_TasksEntry <: ProtoType
    key::Int32
    value::AbstractString
    JobDef_TasksEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct JobDef_TasksEntry (mapentry)

mutable struct JobDef <: ProtoType
    name::AbstractString
    tasks::Base.Dict{Int32,AbstractString} # map entry
    JobDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct JobDef

mutable struct ClusterDef <: ProtoType
    job::Base.Vector{JobDef}
    ClusterDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ClusterDef

export JobDef_TasksEntry, JobDef, ClusterDef
# mapentries: "JobDef_TasksEntry" => ("Int32", "AbstractString")
