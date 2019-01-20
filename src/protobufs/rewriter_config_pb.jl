# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct AutoParallelOptions <: ProtoType
    enable::Bool
    num_replicas::Int32
    AutoParallelOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AutoParallelOptions

mutable struct ScopedAllocatorOptions <: ProtoType
    enable_op::Base.Vector{AbstractString}
    ScopedAllocatorOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ScopedAllocatorOptions

struct __enum_RewriterConfig_Toggle <: ProtoEnum
    DEFAULT::Int32
    ON::Int32
    OFF::Int32
    AGGRESSIVE::Int32
    __enum_RewriterConfig_Toggle() = new(0,1,2,3)
end #struct __enum_RewriterConfig_Toggle
const RewriterConfig_Toggle = __enum_RewriterConfig_Toggle()

struct __enum_RewriterConfig_NumIterationsType <: ProtoEnum
    DEFAULT_NUM_ITERS::Int32
    ONE::Int32
    TWO::Int32
    __enum_RewriterConfig_NumIterationsType() = new(0,1,2)
end #struct __enum_RewriterConfig_NumIterationsType
const RewriterConfig_NumIterationsType = __enum_RewriterConfig_NumIterationsType()

struct __enum_RewriterConfig_MemOptType <: ProtoEnum
    DEFAULT_MEM_OPT::Int32
    NO_MEM_OPT::Int32
    MANUAL::Int32
    SWAPPING_HEURISTICS::Int32
    RECOMPUTATION_HEURISTICS::Int32
    SCHEDULING_HEURISTICS::Int32
    HEURISTICS::Int32
    __enum_RewriterConfig_MemOptType() = new(0,1,2,4,5,6,3)
end #struct __enum_RewriterConfig_MemOptType
const RewriterConfig_MemOptType = __enum_RewriterConfig_MemOptType()

mutable struct RewriterConfig_CustomGraphOptimizer_ParameterMapEntry <: ProtoType
    key::AbstractString
    value::AttrValue
    RewriterConfig_CustomGraphOptimizer_ParameterMapEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RewriterConfig_CustomGraphOptimizer_ParameterMapEntry (mapentry)

mutable struct RewriterConfig_CustomGraphOptimizer <: ProtoType
    name::AbstractString
    parameter_map::Base.Dict{AbstractString,AttrValue} # map entry
    RewriterConfig_CustomGraphOptimizer(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RewriterConfig_CustomGraphOptimizer

mutable struct RewriterConfig <: ProtoType
    layout_optimizer::Int32
    constant_folding::Int32
    shape_optimization::Int32
    remapping::Int32
    arithmetic_optimization::Int32
    dependency_optimization::Int32
    loop_optimization::Int32
    function_optimization::Int32
    debug_stripper::Int32
    disable_model_pruning::Bool
    scoped_allocator_optimization::Int32
    pin_to_host_optimization::Int32
    disable_meta_optimizer::Bool
    meta_optimizer_iterations::Int32
    min_graph_nodes::Int32
    memory_optimization::Int32
    memory_optimizer_target_node_name_scope::AbstractString
    meta_optimizer_timeout_ms::Int64
    auto_parallel::AutoParallelOptions
    fail_on_optimizer_errors::Bool
    scoped_allocator_opts::ScopedAllocatorOptions
    optimizers::Base.Vector{AbstractString}
    custom_optimizers::Base.Vector{RewriterConfig_CustomGraphOptimizer}
    RewriterConfig(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RewriterConfig
const __fnum_RewriterConfig = Int[1,3,13,14,7,8,9,10,11,2,15,18,19,12,17,4,6,20,5,21,16,100,200]
meta(t::Type{RewriterConfig}) = meta(t, ProtoBuf.DEF_REQ, __fnum_RewriterConfig, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

export AutoParallelOptions, ScopedAllocatorOptions, RewriterConfig_Toggle, RewriterConfig_NumIterationsType, RewriterConfig_MemOptType, RewriterConfig_CustomGraphOptimizer_ParameterMapEntry, RewriterConfig_CustomGraphOptimizer, RewriterConfig
# mapentries: "RewriterConfig_CustomGraphOptimizer_ParameterMapEntry" => ("AbstractString", "AttrValue")
