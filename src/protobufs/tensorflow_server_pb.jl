# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct ServerDef <: ProtoType
    cluster::ClusterDef
    job_name::AbstractString
    task_index::Int32
    default_session_config::ConfigProto
    protocol::AbstractString
    ServerDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ServerDef

export ServerDef
