# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type AllocatorMemoryUsed
    allocator_name::AbstractString
    total_bytes::Int64
    peak_bytes::Int64
    AllocatorMemoryUsed(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type AllocatorMemoryUsed
hash(v::AllocatorMemoryUsed) = ProtoBuf.protohash(v)
isequal(v1::AllocatorMemoryUsed, v2::AllocatorMemoryUsed) = ProtoBuf.protoisequal(v1, v2)
==(v1::AllocatorMemoryUsed, v2::AllocatorMemoryUsed) = ProtoBuf.protoeq(v1, v2)

type NodeOutput
    slot::Int32
    tensor_description::TensorDescription
    NodeOutput(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NodeOutput
const __fnum_NodeOutput = Int[1,3]
meta(t::Type{NodeOutput}) = meta(t, ProtoBuf.DEF_REQ, __fnum_NodeOutput, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::NodeOutput) = ProtoBuf.protohash(v)
isequal(v1::NodeOutput, v2::NodeOutput) = ProtoBuf.protoisequal(v1, v2)
==(v1::NodeOutput, v2::NodeOutput) = ProtoBuf.protoeq(v1, v2)

type NodeExecStats
    node_name::AbstractString
    all_start_micros::Int64
    op_start_rel_micros::Int64
    op_end_rel_micros::Int64
    all_end_rel_micros::Int64
    memory::Array{AllocatorMemoryUsed,1}
    output::Array{NodeOutput,1}
    timeline_label::AbstractString
    scheduled_micros::Int64
    thread_id::UInt32
    referenced_tensor::Array{AllocationDescription,1}
    NodeExecStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type NodeExecStats
hash(v::NodeExecStats) = ProtoBuf.protohash(v)
isequal(v1::NodeExecStats, v2::NodeExecStats) = ProtoBuf.protoisequal(v1, v2)
==(v1::NodeExecStats, v2::NodeExecStats) = ProtoBuf.protoeq(v1, v2)

type DeviceStepStats
    device::AbstractString
    node_stats::Array{NodeExecStats,1}
    DeviceStepStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type DeviceStepStats
hash(v::DeviceStepStats) = ProtoBuf.protohash(v)
isequal(v1::DeviceStepStats, v2::DeviceStepStats) = ProtoBuf.protoisequal(v1, v2)
==(v1::DeviceStepStats, v2::DeviceStepStats) = ProtoBuf.protoeq(v1, v2)

type StepStats
    dev_stats::Array{DeviceStepStats,1}
    StepStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type StepStats
hash(v::StepStats) = ProtoBuf.protohash(v)
isequal(v1::StepStats, v2::StepStats) = ProtoBuf.protoisequal(v1, v2)
==(v1::StepStats, v2::StepStats) = ProtoBuf.protoeq(v1, v2)

export AllocatorMemoryUsed, NodeOutput, NodeExecStats, DeviceStepStats, StepStats
