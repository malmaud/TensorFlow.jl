# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct GraphTransferNodeInput <: ProtoType
    node_id::Int32
    output_port::Int32
    GraphTransferNodeInput(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferNodeInput

mutable struct GraphTransferNodeInfo <: ProtoType
    name::AbstractString
    node_id::Int32
    type_name::AbstractString
    soc_op_id::Int32
    padding_id::Int32
    input_count::Int32
    output_count::Int32
    GraphTransferNodeInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferNodeInfo

mutable struct GraphTransferConstNodeInfo <: ProtoType
    name::AbstractString
    node_id::Int32
    shape::Base.Vector{Int64}
    data::Array{UInt8,1}
    dtype::Int32
    GraphTransferConstNodeInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferConstNodeInfo
const __pack_GraphTransferConstNodeInfo = Symbol[:shape]
meta(t::Type{GraphTransferConstNodeInfo}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GraphTransferConstNodeInfo, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GraphTransferNodeInputInfo <: ProtoType
    node_id::Int32
    node_input::Base.Vector{GraphTransferNodeInput}
    GraphTransferNodeInputInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferNodeInputInfo

mutable struct GraphTransferNodeOutputInfo <: ProtoType
    node_id::Int32
    max_byte_size::Base.Vector{Int32}
    GraphTransferNodeOutputInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferNodeOutputInfo
const __pack_GraphTransferNodeOutputInfo = Symbol[:max_byte_size]
meta(t::Type{GraphTransferNodeOutputInfo}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GraphTransferNodeOutputInfo, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GraphTransferGraphInputNodeInfo <: ProtoType
    name::AbstractString
    shape::Base.Vector{Int64}
    dtype::Int32
    GraphTransferGraphInputNodeInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferGraphInputNodeInfo
const __pack_GraphTransferGraphInputNodeInfo = Symbol[:shape]
meta(t::Type{GraphTransferGraphInputNodeInfo}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GraphTransferGraphInputNodeInfo, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct GraphTransferGraphOutputNodeInfo <: ProtoType
    name::AbstractString
    shape::Base.Vector{Int64}
    dtype::Int32
    GraphTransferGraphOutputNodeInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferGraphOutputNodeInfo
const __pack_GraphTransferGraphOutputNodeInfo = Symbol[:shape]
meta(t::Type{GraphTransferGraphOutputNodeInfo}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_GraphTransferGraphOutputNodeInfo, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

struct __enum_GraphTransferInfo_Destination <: ProtoEnum
    NOP::Int32
    HEXAGON::Int32
    __enum_GraphTransferInfo_Destination() = new(0,1)
end #struct __enum_GraphTransferInfo_Destination
const GraphTransferInfo_Destination = __enum_GraphTransferInfo_Destination()

mutable struct GraphTransferInfo <: ProtoType
    node_info::Base.Vector{GraphTransferNodeInfo}
    const_node_info::Base.Vector{GraphTransferConstNodeInfo}
    node_input_info::Base.Vector{GraphTransferNodeInputInfo}
    node_output_info::Base.Vector{GraphTransferNodeOutputInfo}
    graph_input_node_info::Base.Vector{GraphTransferGraphInputNodeInfo}
    graph_output_node_info::Base.Vector{GraphTransferGraphOutputNodeInfo}
    destination::Int32
    GraphTransferInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphTransferInfo

export GraphTransferNodeInput, GraphTransferNodeInfo, GraphTransferConstNodeInfo, GraphTransferNodeInputInfo, GraphTransferNodeOutputInfo, GraphTransferGraphInputNodeInfo, GraphTransferGraphOutputNodeInfo, GraphTransferInfo_Destination, GraphTransferInfo
