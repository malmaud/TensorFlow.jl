# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type AllocationDescription
    requested_bytes::Int64
    allocated_bytes::Int64
    allocator_name::AbstractString
    allocation_id::Int64
    has_single_reference::Bool
    ptr::UInt64
    AllocationDescription(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type AllocationDescription
hash(v::AllocationDescription) = ProtoBuf.protohash(v)
isequal(v1::AllocationDescription, v2::AllocationDescription) = ProtoBuf.protoisequal(v1, v2)
==(v1::AllocationDescription, v2::AllocationDescription) = ProtoBuf.protoeq(v1, v2)

export AllocationDescription
