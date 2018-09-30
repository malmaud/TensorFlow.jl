# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct RemoteFusedGraphExecuteInfo_TensorShapeTypeProto <: ProtoType
    dtype::Int32
    shape::TensorShapeProto
    RemoteFusedGraphExecuteInfo_TensorShapeTypeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RemoteFusedGraphExecuteInfo_TensorShapeTypeProto

mutable struct RemoteFusedGraphExecuteInfo <: ProtoType
    remote_graph::GraphDef
    graph_input_node_name::Base.Vector{AbstractString}
    graph_output_node_name::Base.Vector{AbstractString}
    executor_name::AbstractString
    serialized_executor_parameters::Array{UInt8,1}
    default_graph_input_tensor_shape::Base.Vector{RemoteFusedGraphExecuteInfo_TensorShapeTypeProto}
    default_graph_output_tensor_shape::Base.Vector{RemoteFusedGraphExecuteInfo_TensorShapeTypeProto}
    RemoteFusedGraphExecuteInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RemoteFusedGraphExecuteInfo

export RemoteFusedGraphExecuteInfo_TensorShapeTypeProto, RemoteFusedGraphExecuteInfo
