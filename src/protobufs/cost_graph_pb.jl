# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct CostGraphDef_Node_InputInfo <: ProtoType
    preceding_node::Int32
    preceding_port::Int32
    CostGraphDef_Node_InputInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CostGraphDef_Node_InputInfo

mutable struct CostGraphDef_Node_OutputInfo <: ProtoType
    size::Int64
    alias_input_port::Int64
    shape::TensorShapeProto
    dtype::Int32
    CostGraphDef_Node_OutputInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CostGraphDef_Node_OutputInfo

mutable struct CostGraphDef_Node <: ProtoType
    name::AbstractString
    device::AbstractString
    id::Int32
    input_info::Base.Vector{CostGraphDef_Node_InputInfo}
    output_info::Base.Vector{CostGraphDef_Node_OutputInfo}
    temporary_memory_size::Int64
    persistent_memory_size::Int64
    host_temp_memory_size::Int64
    device_temp_memory_size::Int64
    device_persistent_memory_size::Int64
    compute_cost::Int64
    compute_time::Int64
    memory_time::Int64
    is_final::Bool
    control_input::Base.Vector{Int32}
    inaccurate::Bool
    CostGraphDef_Node(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CostGraphDef_Node
const __fnum_CostGraphDef_Node = Int[1,2,3,4,5,6,12,10,11,16,9,14,15,7,8,17]
const __pack_CostGraphDef_Node = Symbol[:control_input]
meta(t::Type{CostGraphDef_Node}) = meta(t, ProtoBuf.DEF_REQ, __fnum_CostGraphDef_Node, ProtoBuf.DEF_VAL, true, __pack_CostGraphDef_Node, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct CostGraphDef <: ProtoType
    node::Base.Vector{CostGraphDef_Node}
    CostGraphDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CostGraphDef

export CostGraphDef_Node_InputInfo, CostGraphDef_Node_OutputInfo, CostGraphDef_Node, CostGraphDef
