# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct GPUOptions_Experimental_VirtualDevices <: ProtoType
    memory_limit_mb::Base.Vector{Float32}
    GPUOptions_Experimental_VirtualDevices(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GPUOptions_Experimental_VirtualDevices
const __pack_GPUOptions_Experimental_VirtualDevices = Symbol[:memory_limit_mb]
meta(t::Type{GPUOptions_Experimental_VirtualDevices}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GPUOptions_Experimental_VirtualDevices, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GPUOptions_Experimental <: ProtoType
    virtual_devices::Base.Vector{GPUOptions_Experimental_VirtualDevices}
    use_unified_memory::Bool
    num_dev_to_dev_copy_streams::Int32
    collective_ring_order::AbstractString
    GPUOptions_Experimental(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GPUOptions_Experimental

mutable struct GPUOptions <: ProtoType
    per_process_gpu_memory_fraction::Float64
    allow_growth::Bool
    allocator_type::AbstractString
    deferred_deletion_bytes::Int64
    visible_device_list::AbstractString
    polling_active_delay_usecs::Int32
    polling_inactive_delay_msecs::Int32
    force_gpu_compatible::Bool
    experimental::GPUOptions_Experimental
    GPUOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GPUOptions
const __fnum_GPUOptions = Int[1,4,2,3,5,6,7,8,9]
meta(t::Type{GPUOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GPUOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

struct __enum_OptimizerOptions_Level <: ProtoEnum
    L1::Int32
    L0::Int32
    __enum_OptimizerOptions_Level() = new(0,-1)
end #struct __enum_OptimizerOptions_Level
const OptimizerOptions_Level = __enum_OptimizerOptions_Level()

struct __enum_OptimizerOptions_GlobalJitLevel <: ProtoEnum
    DEFAULT::Int32
    OFF::Int32
    ON_1::Int32
    ON_2::Int32
    __enum_OptimizerOptions_GlobalJitLevel() = new(0,-1,1,2)
end #struct __enum_OptimizerOptions_GlobalJitLevel
const OptimizerOptions_GlobalJitLevel = __enum_OptimizerOptions_GlobalJitLevel()

mutable struct OptimizerOptions <: ProtoType
    do_common_subexpression_elimination::Bool
    do_constant_folding::Bool
    max_folded_constant_in_bytes::Int64
    do_function_inlining::Bool
    opt_level::Int32
    global_jit_level::Int32
    OptimizerOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OptimizerOptions
const __fnum_OptimizerOptions = Int[1,2,6,4,3,5]
meta(t::Type{OptimizerOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OptimizerOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GraphOptions <: ProtoType
    enable_recv_scheduling::Bool
    optimizer_options::OptimizerOptions
    build_cost_model::Int64
    build_cost_model_after::Int64
    infer_shapes::Bool
    place_pruned_graph::Bool
    enable_bfloat16_sendrecv::Bool
    timeline_step::Int32
    rewrite_options::RewriterConfig
    GraphOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphOptions
const __fnum_GraphOptions = Int[2,3,4,9,5,6,7,8,10]
meta(t::Type{GraphOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GraphOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ThreadPoolOptionProto <: ProtoType
    num_threads::Int32
    global_name::AbstractString
    ThreadPoolOptionProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ThreadPoolOptionProto

mutable struct RPCOptions <: ProtoType
    use_rpc_for_inprocess_master::Bool
    compression_algorithm::AbstractString
    compression_level::Int32
    RPCOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RPCOptions

mutable struct ConfigProto_DeviceCountEntry <: ProtoType
    key::AbstractString
    value::Int32
    ConfigProto_DeviceCountEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConfigProto_DeviceCountEntry (mapentry)

mutable struct ConfigProto_Experimental <: ProtoType
    collective_group_leader::AbstractString
    executor_type::AbstractString
    recv_buf_max_chunk::Int32
    use_numa_affinity::Bool
    collective_deterministic_sequential_execution::Bool
    ConfigProto_Experimental(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConfigProto_Experimental
const __fnum_ConfigProto_Experimental = Int[1,3,4,5,6]
meta(t::Type{ConfigProto_Experimental}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConfigProto_Experimental, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct ConfigProto <: ProtoType
    device_count::Base.Dict{AbstractString,Int32} # map entry
    intra_op_parallelism_threads::Int32
    inter_op_parallelism_threads::Int32
    use_per_session_threads::Bool
    session_inter_op_thread_pool::Base.Vector{ThreadPoolOptionProto}
    placement_period::Int32
    device_filters::Base.Vector{AbstractString}
    gpu_options::GPUOptions
    allow_soft_placement::Bool
    log_device_placement::Bool
    graph_options::GraphOptions
    operation_timeout_in_ms::Int64
    rpc_options::RPCOptions
    cluster_def::ClusterDef
    isolate_session_state::Bool
    experimental::ConfigProto_Experimental
    ConfigProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ConfigProto
const __fnum_ConfigProto = Int[1,2,5,9,12,3,4,6,7,8,10,11,13,14,15,16]
meta(t::Type{ConfigProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConfigProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

struct __enum_RunOptions_TraceLevel <: ProtoEnum
    NO_TRACE::Int32
    SOFTWARE_TRACE::Int32
    HARDWARE_TRACE::Int32
    FULL_TRACE::Int32
    __enum_RunOptions_TraceLevel() = new(0,1,2,3)
end #struct __enum_RunOptions_TraceLevel
const RunOptions_TraceLevel = __enum_RunOptions_TraceLevel()

mutable struct RunOptions_Experimental <: ProtoType
    collective_graph_key::Int64
    use_run_handler_pool::Bool
    RunOptions_Experimental(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunOptions_Experimental

mutable struct RunOptions <: ProtoType
    trace_level::Int32
    timeout_in_ms::Int64
    inter_op_thread_pool::Int32
    output_partition_graphs::Bool
    debug_options::DebugOptions
    report_tensor_allocations_upon_oom::Bool
    experimental::RunOptions_Experimental
    RunOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunOptions
const __fnum_RunOptions = Int[1,2,3,5,6,7,8]
meta(t::Type{RunOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_RunOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RunMetadata <: ProtoType
    step_stats::StepStats
    cost_graph::CostGraphDef
    partition_graphs::Base.Vector{GraphDef}
    RunMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunMetadata

mutable struct TensorConnection <: ProtoType
    from_tensor::AbstractString
    to_tensor::AbstractString
    TensorConnection(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorConnection

mutable struct CallableOptions_FeedDevicesEntry <: ProtoType
    key::AbstractString
    value::AbstractString
    CallableOptions_FeedDevicesEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CallableOptions_FeedDevicesEntry (mapentry)

mutable struct CallableOptions_FetchDevicesEntry <: ProtoType
    key::AbstractString
    value::AbstractString
    CallableOptions_FetchDevicesEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CallableOptions_FetchDevicesEntry (mapentry)

mutable struct CallableOptions <: ProtoType
    feed::Base.Vector{AbstractString}
    fetch::Base.Vector{AbstractString}
    target::Base.Vector{AbstractString}
    run_options::RunOptions
    tensor_connection::Base.Vector{TensorConnection}
    feed_devices::Base.Dict{AbstractString,AbstractString} # map entry
    fetch_devices::Base.Dict{AbstractString,AbstractString} # map entry
    fetch_skip_sync::Bool
    CallableOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CallableOptions

export GPUOptions_Experimental_VirtualDevices, GPUOptions_Experimental, GPUOptions, OptimizerOptions_Level, OptimizerOptions_GlobalJitLevel, OptimizerOptions, GraphOptions, ThreadPoolOptionProto, RPCOptions, ConfigProto_DeviceCountEntry, ConfigProto_Experimental, ConfigProto, RunOptions_TraceLevel, RunOptions_Experimental, RunOptions, RunMetadata, TensorConnection, CallableOptions_FeedDevicesEntry, CallableOptions_FetchDevicesEntry, CallableOptions
# mapentries: "CallableOptions_FeedDevicesEntry" => ("AbstractString", "AbstractString"), "CallableOptions_FetchDevicesEntry" => ("AbstractString", "AbstractString"), "ConfigProto_DeviceCountEntry" => ("AbstractString", "Int32")
