# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ProtoBuf.google.protobuf

mutable struct CollectionDef_NodeList <: ProtoType
    value::Base.Vector{AbstractString}
    CollectionDef_NodeList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CollectionDef_NodeList

mutable struct CollectionDef_BytesList <: ProtoType
    value::Base.Vector{Array{UInt8,1}}
    CollectionDef_BytesList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CollectionDef_BytesList

mutable struct CollectionDef_Int64List <: ProtoType
    value::Base.Vector{Int64}
    CollectionDef_Int64List(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CollectionDef_Int64List
const __pack_CollectionDef_Int64List = Symbol[:value]
meta(t::Type{CollectionDef_Int64List}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_CollectionDef_Int64List, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct CollectionDef_FloatList <: ProtoType
    value::Base.Vector{Float32}
    CollectionDef_FloatList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CollectionDef_FloatList
const __pack_CollectionDef_FloatList = Symbol[:value]
meta(t::Type{CollectionDef_FloatList}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_CollectionDef_FloatList, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct CollectionDef_AnyList <: ProtoType
    value::Base.Vector{ProtoBuf.google.protobuf._Any}
    CollectionDef_AnyList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CollectionDef_AnyList

mutable struct CollectionDef <: ProtoType
    node_list::CollectionDef_NodeList
    bytes_list::CollectionDef_BytesList
    int64_list::CollectionDef_Int64List
    float_list::CollectionDef_FloatList
    any_list::CollectionDef_AnyList
    CollectionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CollectionDef
const __oneofs_CollectionDef = Int[1,1,1,1,1]
const __oneof_names_CollectionDef = [Symbol("kind")]
meta(t::Type{CollectionDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_CollectionDef, __oneof_names_CollectionDef, ProtoBuf.DEF_FIELD_TYPES)

mutable struct MetaGraphDef_CollectionDefEntry <: ProtoType
    key::AbstractString
    value::CollectionDef
    MetaGraphDef_CollectionDefEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MetaGraphDef_CollectionDefEntry (mapentry)

mutable struct TensorInfo_CooSparse <: ProtoType
    values_tensor_name::AbstractString
    indices_tensor_name::AbstractString
    dense_shape_tensor_name::AbstractString
    TensorInfo_CooSparse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorInfo_CooSparse

mutable struct TensorInfo <: ProtoType
    name::AbstractString
    coo_sparse::TensorInfo_CooSparse
    dtype::Int32
    tensor_shape::TensorShapeProto
    TensorInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorInfo
const __fnum_TensorInfo = Int[1,4,2,3]
const __oneofs_TensorInfo = Int[1,1,0,0]
const __oneof_names_TensorInfo = [Symbol("encoding")]
meta(t::Type{TensorInfo}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorInfo, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_TensorInfo, __oneof_names_TensorInfo, ProtoBuf.DEF_FIELD_TYPES)

mutable struct SignatureDef_InputsEntry <: ProtoType
    key::AbstractString
    value::TensorInfo
    SignatureDef_InputsEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SignatureDef_InputsEntry (mapentry)

mutable struct SignatureDef_OutputsEntry <: ProtoType
    key::AbstractString
    value::TensorInfo
    SignatureDef_OutputsEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SignatureDef_OutputsEntry (mapentry)

mutable struct SignatureDef <: ProtoType
    inputs::Base.Dict{AbstractString,TensorInfo} # map entry
    outputs::Base.Dict{AbstractString,TensorInfo} # map entry
    method_name::AbstractString
    SignatureDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SignatureDef

mutable struct MetaGraphDef_SignatureDefEntry <: ProtoType
    key::AbstractString
    value::SignatureDef
    MetaGraphDef_SignatureDefEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MetaGraphDef_SignatureDefEntry (mapentry)

mutable struct AssetFileDef <: ProtoType
    tensor_info::TensorInfo
    filename::AbstractString
    AssetFileDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AssetFileDef

mutable struct MetaGraphDef_MetaInfoDef <: ProtoType
    meta_graph_version::AbstractString
    stripped_op_list::OpList
    any_info::ProtoBuf.google.protobuf._Any
    tags::Base.Vector{AbstractString}
    tensorflow_version::AbstractString
    tensorflow_git_version::AbstractString
    stripped_default_attrs::Bool
    MetaGraphDef_MetaInfoDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MetaGraphDef_MetaInfoDef

mutable struct MetaGraphDef <: ProtoType
    meta_info_def::MetaGraphDef_MetaInfoDef
    graph_def::GraphDef
    saver_def::SaverDef
    collection_def::Base.Dict{AbstractString,CollectionDef} # map entry
    signature_def::Base.Dict{AbstractString,SignatureDef} # map entry
    asset_file_def::Base.Vector{AssetFileDef}
    MetaGraphDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MetaGraphDef

export MetaGraphDef_MetaInfoDef, MetaGraphDef_CollectionDefEntry, MetaGraphDef_SignatureDefEntry, MetaGraphDef, CollectionDef_NodeList, CollectionDef_BytesList, CollectionDef_Int64List, CollectionDef_FloatList, CollectionDef_AnyList, CollectionDef, TensorInfo_CooSparse, TensorInfo, SignatureDef_InputsEntry, SignatureDef_OutputsEntry, SignatureDef, AssetFileDef
# mapentries: "MetaGraphDef_SignatureDefEntry" => ("AbstractString", "SignatureDef"), "MetaGraphDef_CollectionDefEntry" => ("AbstractString", "CollectionDef"), "SignatureDef_InputsEntry" => ("AbstractString", "TensorInfo"), "SignatureDef_OutputsEntry" => ("AbstractString", "TensorInfo")
