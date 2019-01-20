# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ProtoBuf.google.protobuf
import ._ProtoBuf_Top_.tensorflow

mutable struct GetStatusRequest <: ProtoType
    GetStatusRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetStatusRequest

mutable struct GetStatusResponse <: ProtoType
    device_attributes::Base.Vector{DeviceAttributes}
    GetStatusResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetStatusResponse

mutable struct CreateWorkerSessionRequest <: ProtoType
    session_handle::AbstractString
    server_def::ServerDef
    isolate_session_state::Bool
    CreateWorkerSessionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateWorkerSessionRequest

mutable struct CreateWorkerSessionResponse <: ProtoType
    CreateWorkerSessionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateWorkerSessionResponse

mutable struct DeleteWorkerSessionRequest <: ProtoType
    session_handle::AbstractString
    DeleteWorkerSessionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeleteWorkerSessionRequest

mutable struct DeleteWorkerSessionResponse <: ProtoType
    DeleteWorkerSessionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeleteWorkerSessionResponse

mutable struct RegisterGraphRequest <: ProtoType
    session_handle::AbstractString
    create_worker_session_called::Bool
    graph_def::GraphDef
    has_control_flow::Bool
    graph_options::GraphOptions
    debug_options::DebugOptions
    collective_graph_key::Int64
    RegisterGraphRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RegisterGraphRequest
const __fnum_RegisterGraphRequest = Int[1,6,2,3,4,5,7]
meta(t::Type{RegisterGraphRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_RegisterGraphRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RegisterGraphResponse <: ProtoType
    graph_handle::AbstractString
    RegisterGraphResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RegisterGraphResponse

mutable struct DeregisterGraphRequest <: ProtoType
    session_handle::AbstractString
    create_worker_session_called::Bool
    graph_handle::AbstractString
    DeregisterGraphRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeregisterGraphRequest
const __fnum_DeregisterGraphRequest = Int[2,3,1]
meta(t::Type{DeregisterGraphRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DeregisterGraphRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DeregisterGraphResponse <: ProtoType
    DeregisterGraphResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DeregisterGraphResponse

mutable struct CleanupAllRequest <: ProtoType
    container::Base.Vector{AbstractString}
    CleanupAllRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CleanupAllRequest

mutable struct CleanupAllResponse <: ProtoType
    CleanupAllResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CleanupAllResponse

mutable struct ExecutorOpts <: ProtoType
    record_costs::Bool
    record_timeline::Bool
    record_partition_graphs::Bool
    report_tensor_allocations_upon_oom::Bool
    ExecutorOpts(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExecutorOpts
const __fnum_ExecutorOpts = Int[1,3,4,5]
meta(t::Type{ExecutorOpts}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ExecutorOpts, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RunGraphRequest <: ProtoType
    session_handle::AbstractString
    create_worker_session_called::Bool
    graph_handle::AbstractString
    step_id::Int64
    exec_opts::ExecutorOpts
    send::Base.Vector{NamedTensorProto}
    recv_key::Base.Vector{AbstractString}
    is_partial::Bool
    is_last_partial_run::Bool
    store_errors_in_response_body::Bool
    RunGraphRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunGraphRequest
const __fnum_RunGraphRequest = Int[8,10,1,2,5,3,4,6,7,9]
meta(t::Type{RunGraphRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_RunGraphRequest, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RunGraphResponse <: ProtoType
    recv::Base.Vector{NamedTensorProto}
    step_stats::StepStats
    cost_graph::CostGraphDef
    partition_graph::Base.Vector{GraphDef}
    status_code::Int32
    status_error_message::AbstractString
    RunGraphResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunGraphResponse

mutable struct CleanupGraphRequest <: ProtoType
    step_id::Int64
    CleanupGraphRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CleanupGraphRequest

mutable struct CleanupGraphResponse <: ProtoType
    CleanupGraphResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CleanupGraphResponse

mutable struct RecvTensorRequest <: ProtoType
    step_id::Int64
    rendezvous_key::AbstractString
    dma_ok::Bool
    client_locality::DeviceLocality
    server_locality::DeviceLocality
    transport_options::ProtoBuf.google.protobuf._Any
    request_id::Int64
    RecvTensorRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RecvTensorRequest

mutable struct RecvTensorResponse <: ProtoType
    tensor::TensorProto
    is_dead::Bool
    send_start_micros::Int64
    transport_options::ProtoBuf.google.protobuf._Any
    RecvTensorResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RecvTensorResponse

mutable struct LoggingRequest <: ProtoType
    enable_rpc_logging::Bool
    disable_rpc_logging::Bool
    clear::Bool
    fetch_step_id::Base.Vector{Int64}
    LoggingRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LoggingRequest
const __fnum_LoggingRequest = Int[1,4,2,3]
const __pack_LoggingRequest = Symbol[:fetch_step_id]
meta(t::Type{LoggingRequest}) = meta(t, ProtoBuf.DEF_REQ, __fnum_LoggingRequest, ProtoBuf.DEF_VAL, true, __pack_LoggingRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct LabeledStepStats <: ProtoType
    step_id::Int64
    step_stats::StepStats
    LabeledStepStats(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LabeledStepStats

mutable struct LoggingResponse <: ProtoType
    step::Base.Vector{LabeledStepStats}
    LoggingResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct LoggingResponse

mutable struct TraceOpts <: ProtoType
    duration::Float64
    use_step_profiler::Bool
    use_kernel_profiler::Bool
    use_extended_profiler::Bool
    use_gpu_profiler::Bool
    use_sample_profiler::Bool
    TraceOpts(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TraceOpts

mutable struct TracingRequest <: ProtoType
    options::TraceOpts
    TracingRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TracingRequest

mutable struct TracingResponse <: ProtoType
    TracingResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TracingResponse

mutable struct RecvBufRequest <: ProtoType
    step_id::Int64
    buf_rendezvous_key::AbstractString
    num_bytes::Int64
    buf_ptr::UInt64
    client_locality::DeviceLocality
    server_locality::DeviceLocality
    transport_options::ProtoBuf.google.protobuf._Any
    src_device::AbstractString
    dst_device::AbstractString
    request_id::Int64
    RecvBufRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RecvBufRequest
const __wtype_RecvBufRequest = Dict(:buf_ptr => :fixed64)
meta(t::Type{RecvBufRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_RecvBufRequest, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RecvBufResponse <: ProtoType
    buf_ptr::UInt64
    num_bytes::Int64
    is_dead::Bool
    transport_options::ProtoBuf.google.protobuf._Any
    send_start_micros::Int64
    RecvBufResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RecvBufResponse
const __wtype_RecvBufResponse = Dict(:buf_ptr => :fixed64)
meta(t::Type{RecvBufResponse}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, __wtype_RecvBufResponse, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct CompleteGroupRequest <: ProtoType
    group_key::Int32
    group_size::Int32
    device_type::AbstractString
    device_name::Base.Vector{AbstractString}
    CompleteGroupRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CompleteGroupRequest

mutable struct CompleteGroupResponse <: ProtoType
    group_key::Int32
    group_size::Int32
    device_type::AbstractString
    num_tasks::Int32
    device_name::Base.Vector{AbstractString}
    task_name::Base.Vector{AbstractString}
    CompleteGroupResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CompleteGroupResponse

mutable struct CompleteInstanceRequest <: ProtoType
    name::AbstractString
    _type::Int32
    data_type::Int32
    shape::TensorShapeProto
    group_key::Int32
    group_size::Int32
    instance_key::Int32
    device_type::AbstractString
    subdiv_offset::Base.Vector{Int32}
    device::AbstractString
    is_source::Bool
    CompleteInstanceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CompleteInstanceRequest
const __pack_CompleteInstanceRequest = Symbol[:subdiv_offset]
meta(t::Type{CompleteInstanceRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_CompleteInstanceRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct CompleteInstanceResponse <: ProtoType
    instance_key::Int32
    source_rank::Int32
    communicator_key::Array{UInt8,1}
    CompleteInstanceResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CompleteInstanceResponse

mutable struct GetStepSequenceRequest <: ProtoType
    graph_key::Base.Vector{Int64}
    GetStepSequenceRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetStepSequenceRequest
const __pack_GetStepSequenceRequest = Symbol[:graph_key]
meta(t::Type{GetStepSequenceRequest}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GetStepSequenceRequest, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct StepSequence <: ProtoType
    graph_key::Int64
    next_step_id::Int64
    StepSequence(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct StepSequence

mutable struct GetStepSequenceResponse <: ProtoType
    step_sequence::Base.Vector{StepSequence}
    GetStepSequenceResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GetStepSequenceResponse

export GetStatusRequest, GetStatusResponse, CreateWorkerSessionRequest, CreateWorkerSessionResponse, DeleteWorkerSessionRequest, DeleteWorkerSessionResponse, RegisterGraphRequest, RegisterGraphResponse, DeregisterGraphRequest, DeregisterGraphResponse, CleanupAllRequest, CleanupAllResponse, ExecutorOpts, RunGraphRequest, RunGraphResponse, CleanupGraphRequest, CleanupGraphResponse, RecvTensorRequest, RecvTensorResponse, LoggingRequest, LabeledStepStats, LoggingResponse, TraceOpts, TracingRequest, TracingResponse, RecvBufRequest, RecvBufResponse, CompleteGroupRequest, CompleteGroupResponse, CompleteInstanceRequest, CompleteInstanceResponse, GetStepSequenceRequest, StepSequence, GetStepSequenceResponse
