# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct AllocationDescription <: ProtoType
    requested_bytes::Int64
    allocated_bytes::Int64
    allocator_name::AbstractString
    allocation_id::Int64
    has_single_reference::Bool
    ptr::UInt64
    AllocationDescription(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AllocationDescription

export AllocationDescription
