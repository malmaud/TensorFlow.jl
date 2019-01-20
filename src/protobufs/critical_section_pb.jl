# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct CriticalSectionDef <: ProtoType
    critical_section_name::AbstractString
    CriticalSectionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CriticalSectionDef

mutable struct CriticalSectionExecutionDef <: ProtoType
    execute_in_critical_section_name::AbstractString
    exclusive_resource_access::Bool
    CriticalSectionExecutionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CriticalSectionExecutionDef

export CriticalSectionDef, CriticalSectionExecutionDef
