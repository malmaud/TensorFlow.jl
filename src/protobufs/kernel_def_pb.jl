# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct KernelDef_AttrConstraint <: ProtoType
    name::AbstractString
    allowed_values::AttrValue
    KernelDef_AttrConstraint(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct KernelDef_AttrConstraint

mutable struct KernelDef <: ProtoType
    op::AbstractString
    device_type::AbstractString
    constraint::Base.Vector{KernelDef_AttrConstraint}
    host_memory_arg::Base.Vector{AbstractString}
    label::AbstractString
    KernelDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct KernelDef

mutable struct KernelList <: ProtoType
    kernel::Base.Vector{KernelDef}
    KernelList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct KernelList

export KernelDef_AttrConstraint, KernelDef, KernelList
