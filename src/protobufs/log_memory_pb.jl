# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct MemoryLogStep <: ProtoType
    step_id::Int64
    handle::AbstractString
    MemoryLogStep(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryLogStep

mutable struct MemoryLogTensorAllocation <: ProtoType
    step_id::Int64
    kernel_name::AbstractString
    tensor::TensorDescription
    MemoryLogTensorAllocation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryLogTensorAllocation

mutable struct MemoryLogTensorDeallocation <: ProtoType
    allocation_id::Int64
    allocator_name::AbstractString
    MemoryLogTensorDeallocation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryLogTensorDeallocation

mutable struct MemoryLogTensorOutput <: ProtoType
    step_id::Int64
    kernel_name::AbstractString
    index::Int32
    tensor::TensorDescription
    MemoryLogTensorOutput(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryLogTensorOutput

mutable struct MemoryLogRawAllocation <: ProtoType
    step_id::Int64
    operation::AbstractString
    num_bytes::Int64
    ptr::UInt64
    allocation_id::Int64
    allocator_name::AbstractString
    MemoryLogRawAllocation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryLogRawAllocation

mutable struct MemoryLogRawDeallocation <: ProtoType
    step_id::Int64
    operation::AbstractString
    allocation_id::Int64
    allocator_name::AbstractString
    deferred::Bool
    MemoryLogRawDeallocation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryLogRawDeallocation

export MemoryLogStep, MemoryLogTensorAllocation, MemoryLogTensorDeallocation, MemoryLogTensorOutput, MemoryLogRawAllocation, MemoryLogRawDeallocation
