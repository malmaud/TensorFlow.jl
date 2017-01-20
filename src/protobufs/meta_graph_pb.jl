# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==
using ProtoBuf.google.protobuf

type CollectionDef_NodeList
    value::Array{AbstractString,1}
    CollectionDef_NodeList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CollectionDef_NodeList
hash(v::CollectionDef_NodeList) = ProtoBuf.protohash(v)
isequal(v1::CollectionDef_NodeList, v2::CollectionDef_NodeList) = ProtoBuf.protoisequal(v1, v2)
==(v1::CollectionDef_NodeList, v2::CollectionDef_NodeList) = ProtoBuf.protoeq(v1, v2)

type CollectionDef_BytesList
    value::Array{Array{UInt8,1},1}
    CollectionDef_BytesList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CollectionDef_BytesList
hash(v::CollectionDef_BytesList) = ProtoBuf.protohash(v)
isequal(v1::CollectionDef_BytesList, v2::CollectionDef_BytesList) = ProtoBuf.protoisequal(v1, v2)
==(v1::CollectionDef_BytesList, v2::CollectionDef_BytesList) = ProtoBuf.protoeq(v1, v2)

type CollectionDef_Int64List
    value::Array{Int64,1}
    CollectionDef_Int64List(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CollectionDef_Int64List
const __pack_CollectionDef_Int64List = Symbol[:value]
meta(t::Type{CollectionDef_Int64List}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_CollectionDef_Int64List, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::CollectionDef_Int64List) = ProtoBuf.protohash(v)
isequal(v1::CollectionDef_Int64List, v2::CollectionDef_Int64List) = ProtoBuf.protoisequal(v1, v2)
==(v1::CollectionDef_Int64List, v2::CollectionDef_Int64List) = ProtoBuf.protoeq(v1, v2)

type CollectionDef_FloatList
    value::Array{Float32,1}
    CollectionDef_FloatList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CollectionDef_FloatList
const __pack_CollectionDef_FloatList = Symbol[:value]
meta(t::Type{CollectionDef_FloatList}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_CollectionDef_FloatList, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::CollectionDef_FloatList) = ProtoBuf.protohash(v)
isequal(v1::CollectionDef_FloatList, v2::CollectionDef_FloatList) = ProtoBuf.protoisequal(v1, v2)
==(v1::CollectionDef_FloatList, v2::CollectionDef_FloatList) = ProtoBuf.protoeq(v1, v2)

type CollectionDef_AnyList
    value::Array{_Any,1}
    CollectionDef_AnyList(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CollectionDef_AnyList
hash(v::CollectionDef_AnyList) = ProtoBuf.protohash(v)
isequal(v1::CollectionDef_AnyList, v2::CollectionDef_AnyList) = ProtoBuf.protoisequal(v1, v2)
==(v1::CollectionDef_AnyList, v2::CollectionDef_AnyList) = ProtoBuf.protoeq(v1, v2)

type CollectionDef
    node_list::CollectionDef_NodeList
    bytes_list::CollectionDef_BytesList
    int64_list::CollectionDef_Int64List
    float_list::CollectionDef_FloatList
    any_list::CollectionDef_AnyList
    CollectionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type CollectionDef
const __oneofs_CollectionDef = Int[1,1,1,1,1]
const __oneof_names_CollectionDef = [@compat(Symbol("kind"))]
meta(t::Type{CollectionDef}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_CollectionDef, __oneof_names_CollectionDef)
hash(v::CollectionDef) = ProtoBuf.protohash(v)
isequal(v1::CollectionDef, v2::CollectionDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::CollectionDef, v2::CollectionDef) = ProtoBuf.protoeq(v1, v2)

type MetaGraphDef_CollectionDefEntry
    key::AbstractString
    value::CollectionDef
    MetaGraphDef_CollectionDefEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type MetaGraphDef_CollectionDefEntry (mapentry)
hash(v::MetaGraphDef_CollectionDefEntry) = ProtoBuf.protohash(v)
isequal(v1::MetaGraphDef_CollectionDefEntry, v2::MetaGraphDef_CollectionDefEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::MetaGraphDef_CollectionDefEntry, v2::MetaGraphDef_CollectionDefEntry) = ProtoBuf.protoeq(v1, v2)

type TensorInfo
    name::AbstractString
    dtype::Int32
    tensor_shape::TensorShapeProto
    TensorInfo(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type TensorInfo
hash(v::TensorInfo) = ProtoBuf.protohash(v)
isequal(v1::TensorInfo, v2::TensorInfo) = ProtoBuf.protoisequal(v1, v2)
==(v1::TensorInfo, v2::TensorInfo) = ProtoBuf.protoeq(v1, v2)

type SignatureDef_InputsEntry
    key::AbstractString
    value::TensorInfo
    SignatureDef_InputsEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SignatureDef_InputsEntry (mapentry)
hash(v::SignatureDef_InputsEntry) = ProtoBuf.protohash(v)
isequal(v1::SignatureDef_InputsEntry, v2::SignatureDef_InputsEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::SignatureDef_InputsEntry, v2::SignatureDef_InputsEntry) = ProtoBuf.protoeq(v1, v2)

type SignatureDef_OutputsEntry
    key::AbstractString
    value::TensorInfo
    SignatureDef_OutputsEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SignatureDef_OutputsEntry (mapentry)
hash(v::SignatureDef_OutputsEntry) = ProtoBuf.protohash(v)
isequal(v1::SignatureDef_OutputsEntry, v2::SignatureDef_OutputsEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::SignatureDef_OutputsEntry, v2::SignatureDef_OutputsEntry) = ProtoBuf.protoeq(v1, v2)

type SignatureDef
    inputs::Dict{AbstractString,TensorInfo} # map entry
    outputs::Dict{AbstractString,TensorInfo} # map entry
    method_name::AbstractString
    SignatureDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SignatureDef
hash(v::SignatureDef) = ProtoBuf.protohash(v)
isequal(v1::SignatureDef, v2::SignatureDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::SignatureDef, v2::SignatureDef) = ProtoBuf.protoeq(v1, v2)

type MetaGraphDef_SignatureDefEntry
    key::AbstractString
    value::SignatureDef
    MetaGraphDef_SignatureDefEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type MetaGraphDef_SignatureDefEntry (mapentry)
hash(v::MetaGraphDef_SignatureDefEntry) = ProtoBuf.protohash(v)
isequal(v1::MetaGraphDef_SignatureDefEntry, v2::MetaGraphDef_SignatureDefEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::MetaGraphDef_SignatureDefEntry, v2::MetaGraphDef_SignatureDefEntry) = ProtoBuf.protoeq(v1, v2)

type AssetFileDef
    tensor_info::TensorInfo
    filename::AbstractString
    AssetFileDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type AssetFileDef
hash(v::AssetFileDef) = ProtoBuf.protohash(v)
isequal(v1::AssetFileDef, v2::AssetFileDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::AssetFileDef, v2::AssetFileDef) = ProtoBuf.protoeq(v1, v2)

type MetaGraphDef_MetaInfoDef
    meta_graph_version::AbstractString
    stripped_op_list::OpList
    any_info::_Any
    tags::Array{AbstractString,1}
    tensorflow_version::AbstractString
    tensorflow_git_version::AbstractString
    MetaGraphDef_MetaInfoDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type MetaGraphDef_MetaInfoDef
hash(v::MetaGraphDef_MetaInfoDef) = ProtoBuf.protohash(v)
isequal(v1::MetaGraphDef_MetaInfoDef, v2::MetaGraphDef_MetaInfoDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::MetaGraphDef_MetaInfoDef, v2::MetaGraphDef_MetaInfoDef) = ProtoBuf.protoeq(v1, v2)

type MetaGraphDef
    meta_info_def::MetaGraphDef_MetaInfoDef
    graph_def::GraphDef
    saver_def::SaverDef
    collection_def::Dict{AbstractString,CollectionDef} # map entry
    signature_def::Dict{AbstractString,SignatureDef} # map entry
    asset_file_def::Array{AssetFileDef,1}
    MetaGraphDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type MetaGraphDef
hash(v::MetaGraphDef) = ProtoBuf.protohash(v)
isequal(v1::MetaGraphDef, v2::MetaGraphDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::MetaGraphDef, v2::MetaGraphDef) = ProtoBuf.protoeq(v1, v2)

export MetaGraphDef_MetaInfoDef, MetaGraphDef_CollectionDefEntry, MetaGraphDef_SignatureDefEntry, MetaGraphDef, CollectionDef_NodeList, CollectionDef_BytesList, CollectionDef_Int64List, CollectionDef_FloatList, CollectionDef_AnyList, CollectionDef, TensorInfo, SignatureDef_InputsEntry, SignatureDef_OutputsEntry, SignatureDef, AssetFileDef
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("MetaGraphDef_SignatureDefEntry",("AbstractString","SignatureDef")), Pair{AbstractString,Tuple{AbstractString,AbstractString}}("MetaGraphDef_CollectionDefEntry",("AbstractString","CollectionDef")), Pair{AbstractString,Tuple{AbstractString,AbstractString}}("SignatureDef_InputsEntry",("AbstractString","TensorInfo")), Pair{AbstractString,Tuple{AbstractString,AbstractString}}("SignatureDef_OutputsEntry",("AbstractString","TensorInfo"))
