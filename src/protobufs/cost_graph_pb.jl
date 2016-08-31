# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type CostGraphDef_Node_InputInfo
    preceding_node::Int32
    preceding_port::Int32
    CostGraphDef_Node_InputInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CostGraphDef_Node_InputInfo
hash(v::CostGraphDef_Node_InputInfo) = ProtoBuf.protohash(v)
isequal(v1::CostGraphDef_Node_InputInfo, v2::CostGraphDef_Node_InputInfo) = ProtoBuf.protoisequal(v1, v2)
==(v1::CostGraphDef_Node_InputInfo, v2::CostGraphDef_Node_InputInfo) = ProtoBuf.protoeq(v1, v2)

type CostGraphDef_Node_OutputInfo
    size::Int64
    alias_input_port::Int64
    CostGraphDef_Node_OutputInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CostGraphDef_Node_OutputInfo
hash(v::CostGraphDef_Node_OutputInfo) = ProtoBuf.protohash(v)
isequal(v1::CostGraphDef_Node_OutputInfo, v2::CostGraphDef_Node_OutputInfo) = ProtoBuf.protoisequal(v1, v2)
==(v1::CostGraphDef_Node_OutputInfo, v2::CostGraphDef_Node_OutputInfo) = ProtoBuf.protoeq(v1, v2)

type CostGraphDef_Node
    name::AbstractString
    device::AbstractString
    id::Int32
    input_info::Array{CostGraphDef_Node_InputInfo,1}
    output_info::Array{CostGraphDef_Node_OutputInfo,1}
    temporary_memory_size::Int64
    is_final::Bool
    control_input::Array{Int32,1}
    CostGraphDef_Node(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CostGraphDef_Node
const __pack_CostGraphDef_Node = Symbol[:control_input]
meta(t::Type{CostGraphDef_Node}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_CostGraphDef_Node, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::CostGraphDef_Node) = ProtoBuf.protohash(v)
isequal(v1::CostGraphDef_Node, v2::CostGraphDef_Node) = ProtoBuf.protoisequal(v1, v2)
==(v1::CostGraphDef_Node, v2::CostGraphDef_Node) = ProtoBuf.protoeq(v1, v2)

type CostGraphDef
    node::Array{CostGraphDef_Node,1}
    CostGraphDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CostGraphDef
hash(v::CostGraphDef) = ProtoBuf.protohash(v)
isequal(v1::CostGraphDef, v2::CostGraphDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::CostGraphDef, v2::CostGraphDef) = ProtoBuf.protoeq(v1, v2)

export CostGraphDef_Node_InputInfo, CostGraphDef_Node_OutputInfo, CostGraphDef_Node, CostGraphDef
