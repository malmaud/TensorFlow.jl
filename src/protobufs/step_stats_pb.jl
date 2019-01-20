# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct AllocationRecord <: ProtoType
    alloc_micros::Int64
    alloc_bytes::Int64
    AllocationRecord(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AllocationRecord

mutable struct AllocatorMemoryUsed <: ProtoType
    allocator_name::AbstractString
    total_bytes::Int64
    peak_bytes::Int64
    live_bytes::Int64
    allocation_records::Base.Vector{AllocationRecord}
    allocator_bytes_in_use::Int64
    AllocatorMemoryUsed(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AllocatorMemoryUsed
const __fnum_AllocatorMemoryUsed = Int[1,2,3,4,6,5]
meta(t::Type{AllocatorMemoryUsed}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AllocatorMemoryUsed, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct NodeOutput <: ProtoType
    slot::Int32
    tensor_description::TensorDescription
    NodeOutput(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NodeOutput
const __fnum_NodeOutput = Int[1,3]
meta(t::Type{NodeOutput}) = meta(t, ProtoBuf.DEF_REQ, __fnum_NodeOutput, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct MemoryStats <: ProtoType
    temp_memory_size::Int64
    persistent_memory_size::Int64
    persistent_tensor_alloc_ids::Base.Vector{Int64}
    device_temp_memory_size::Int64
    device_persistent_memory_size::Int64
    device_persistent_tensor_alloc_ids::Base.Vector{Int64}
    MemoryStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MemoryStats
const __fnum_MemoryStats = Int[1,3,5,2,4,6]
const __pack_MemoryStats = Symbol[:persistent_tensor_alloc_ids,:device_persistent_tensor_alloc_ids]
meta(t::Type{MemoryStats}) = meta(t, ProtoBuf.DEF_REQ, __fnum_MemoryStats, ProtoBuf.DEF_VAL, true, __pack_MemoryStats, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct NodeExecStats <: ProtoType
    node_name::AbstractString
    all_start_micros::Int64
    op_start_rel_micros::Int64
    op_end_rel_micros::Int64
    all_end_rel_micros::Int64
    memory::Base.Vector{AllocatorMemoryUsed}
    output::Base.Vector{NodeOutput}
    timeline_label::AbstractString
    scheduled_micros::Int64
    thread_id::UInt32
    referenced_tensor::Base.Vector{AllocationDescription}
    memory_stats::MemoryStats
    all_start_nanos::Int64
    op_start_rel_nanos::Int64
    op_end_rel_nanos::Int64
    all_end_rel_nanos::Int64
    scheduled_nanos::Int64
    NodeExecStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NodeExecStats

mutable struct DeviceStepStats <: ProtoType
    device::AbstractString
    node_stats::Base.Vector{NodeExecStats}
    DeviceStepStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeviceStepStats

mutable struct StepStats <: ProtoType
    dev_stats::Base.Vector{DeviceStepStats}
    StepStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct StepStats

export AllocationRecord, AllocatorMemoryUsed, NodeOutput, MemoryStats, NodeExecStats, DeviceStepStats, StepStats
