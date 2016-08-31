# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type GPUOptions
    per_process_gpu_memory_fraction::Float64
    allocator_type::AbstractString
    deferred_deletion_bytes::Int64
    allow_growth::Bool
    visible_device_list::AbstractString
    GPUOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type GPUOptions
hash(v::GPUOptions) = ProtoBuf.protohash(v)
isequal(v1::GPUOptions, v2::GPUOptions) = ProtoBuf.protoisequal(v1, v2)
==(v1::GPUOptions, v2::GPUOptions) = ProtoBuf.protoeq(v1, v2)

type __enum_OptimizerOptions_Level <: ProtoEnum
    L1::Int32
    L0::Int32
    __enum_OptimizerOptions_Level() = new(0,0)
end #type __enum_OptimizerOptions_Level
const OptimizerOptions_Level = __enum_OptimizerOptions_Level()

type OptimizerOptions
    do_common_subexpression_elimination::Bool
    do_constant_folding::Bool
    do_function_inlining::Bool
    opt_level::Int32
    OptimizerOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type OptimizerOptions
const __fnum_OptimizerOptions = Int[1,2,4,3]
meta(t::Type{OptimizerOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_OptimizerOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::OptimizerOptions) = ProtoBuf.protohash(v)
isequal(v1::OptimizerOptions, v2::OptimizerOptions) = ProtoBuf.protoisequal(v1, v2)
==(v1::OptimizerOptions, v2::OptimizerOptions) = ProtoBuf.protoeq(v1, v2)

type GraphOptions
    enable_recv_scheduling::Bool
    optimizer_options::OptimizerOptions
    build_cost_model::Int64
    infer_shapes::Bool
    place_pruned_graph::Bool
    GraphOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type GraphOptions
const __fnum_GraphOptions = Int[2,3,4,5,6]
meta(t::Type{GraphOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GraphOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::GraphOptions) = ProtoBuf.protohash(v)
isequal(v1::GraphOptions, v2::GraphOptions) = ProtoBuf.protoisequal(v1, v2)
==(v1::GraphOptions, v2::GraphOptions) = ProtoBuf.protoeq(v1, v2)

type ThreadPoolOptionProto
    num_threads::Int32
    ThreadPoolOptionProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ThreadPoolOptionProto
hash(v::ThreadPoolOptionProto) = ProtoBuf.protohash(v)
isequal(v1::ThreadPoolOptionProto, v2::ThreadPoolOptionProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::ThreadPoolOptionProto, v2::ThreadPoolOptionProto) = ProtoBuf.protoeq(v1, v2)

type ConfigProto_DeviceCountEntry
    key::AbstractString
    value::Int32
    ConfigProto_DeviceCountEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ConfigProto_DeviceCountEntry (mapentry)
hash(v::ConfigProto_DeviceCountEntry) = ProtoBuf.protohash(v)
isequal(v1::ConfigProto_DeviceCountEntry, v2::ConfigProto_DeviceCountEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConfigProto_DeviceCountEntry, v2::ConfigProto_DeviceCountEntry) = ProtoBuf.protoeq(v1, v2)

type ConfigProto
    device_count::Dict{AbstractString,Int32} # map entry
    intra_op_parallelism_threads::Int32
    inter_op_parallelism_threads::Int32
    use_per_session_threads::Bool
    session_inter_op_thread_pool::Array{ThreadPoolOptionProto,1}
    placement_period::Int32
    device_filters::Array{AbstractString,1}
    gpu_options::GPUOptions
    allow_soft_placement::Bool
    log_device_placement::Bool
    graph_options::GraphOptions
    operation_timeout_in_ms::Int64
    ConfigProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type ConfigProto
const __fnum_ConfigProto = Int[1,2,5,9,12,3,4,6,7,8,10,11]
meta(t::Type{ConfigProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ConfigProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::ConfigProto) = ProtoBuf.protohash(v)
isequal(v1::ConfigProto, v2::ConfigProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::ConfigProto, v2::ConfigProto) = ProtoBuf.protoeq(v1, v2)

type DebugTensorWatch
    node_name::AbstractString
    output_slot::Int32
    debug_ops::Array{AbstractString,1}
    debug_urls::Array{AbstractString,1}
    DebugTensorWatch(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type DebugTensorWatch
hash(v::DebugTensorWatch) = ProtoBuf.protohash(v)
isequal(v1::DebugTensorWatch, v2::DebugTensorWatch) = ProtoBuf.protoisequal(v1, v2)
==(v1::DebugTensorWatch, v2::DebugTensorWatch) = ProtoBuf.protoeq(v1, v2)

type __enum_RunOptions_TraceLevel <: ProtoEnum
    NO_TRACE::Int32
    SOFTWARE_TRACE::Int32
    HARDWARE_TRACE::Int32
    FULL_TRACE::Int32
    __enum_RunOptions_TraceLevel() = new(0,1,2,3)
end #type __enum_RunOptions_TraceLevel
const RunOptions_TraceLevel = __enum_RunOptions_TraceLevel()

type RunOptions
    trace_level::Int32
    timeout_in_ms::Int64
    inter_op_thread_pool::Int32
    debug_tensor_watch_opts::Array{DebugTensorWatch,1}
    output_partition_graphs::Bool
    RunOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type RunOptions
hash(v::RunOptions) = ProtoBuf.protohash(v)
isequal(v1::RunOptions, v2::RunOptions) = ProtoBuf.protoisequal(v1, v2)
==(v1::RunOptions, v2::RunOptions) = ProtoBuf.protoeq(v1, v2)

type RunMetadata
    step_stats::StepStats
    cost_graph::CostGraphDef
    partition_graphs::Array{GraphDef,1}
    RunMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type RunMetadata
hash(v::RunMetadata) = ProtoBuf.protohash(v)
isequal(v1::RunMetadata, v2::RunMetadata) = ProtoBuf.protoisequal(v1, v2)
==(v1::RunMetadata, v2::RunMetadata) = ProtoBuf.protoeq(v1, v2)

export GPUOptions, OptimizerOptions_Level, OptimizerOptions, GraphOptions, ThreadPoolOptionProto, ConfigProto_DeviceCountEntry, ConfigProto, DebugTensorWatch, RunOptions_TraceLevel, RunOptions, RunMetadata
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("ConfigProto_DeviceCountEntry",("AbstractString","Int32"))
