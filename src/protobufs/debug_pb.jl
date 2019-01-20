# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct DebugTensorWatch <: ProtoType
    node_name::AbstractString
    output_slot::Int32
    debug_ops::Base.Vector{AbstractString}
    debug_urls::Base.Vector{AbstractString}
    tolerate_debug_op_creation_failures::Bool
    DebugTensorWatch(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DebugTensorWatch

mutable struct DebugOptions <: ProtoType
    debug_tensor_watch_opts::Base.Vector{DebugTensorWatch}
    global_step::Int64
    reset_disk_byte_usage::Bool
    DebugOptions(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DebugOptions
const __fnum_DebugOptions = Int[4,10,11]
meta(t::Type{DebugOptions}) = meta(t, ProtoBuf.DEF_REQ, __fnum_DebugOptions, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct DebuggedSourceFile <: ProtoType
    host::AbstractString
    file_path::AbstractString
    last_modified::Int64
    bytes::Int64
    lines::Base.Vector{AbstractString}
    DebuggedSourceFile(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DebuggedSourceFile

mutable struct DebuggedSourceFiles <: ProtoType
    source_files::Base.Vector{DebuggedSourceFile}
    DebuggedSourceFiles(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct DebuggedSourceFiles

export DebugTensorWatch, DebugOptions, DebuggedSourceFile, DebuggedSourceFiles
