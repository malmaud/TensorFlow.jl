# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct DeviceProperties_EnvironmentEntry <: ProtoType
    key::AbstractString
    value::AbstractString
    DeviceProperties_EnvironmentEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceProperties_EnvironmentEntry (mapentry)

mutable struct DeviceProperties <: ProtoType
    _type::AbstractString
    vendor::AbstractString
    model::AbstractString
    frequency::Int64
    num_cores::Int64
    environment::Base.Dict{AbstractString,AbstractString} # map entry
    num_registers::Int64
    l1_cache_size::Int64
    l2_cache_size::Int64
    l3_cache_size::Int64
    shared_memory_size_per_multiprocessor::Int64
    memory_size::Int64
    bandwidth::Int64
    DeviceProperties(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceProperties

mutable struct NamedDevice <: ProtoType
    name::AbstractString
    properties::DeviceProperties
    NamedDevice(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NamedDevice

export DeviceProperties_EnvironmentEntry, DeviceProperties, NamedDevice
# mapentries: "DeviceProperties_EnvironmentEntry" => ("AbstractString", "AbstractString")
