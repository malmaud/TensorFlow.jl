# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

struct __enum_SaverDef_CheckpointFormatVersion <: ProtoEnum
    LEGACY::Int32
    V1::Int32
    V2::Int32
    __enum_SaverDef_CheckpointFormatVersion() = new(0,1,2)
end #struct __enum_SaverDef_CheckpointFormatVersion
const SaverDef_CheckpointFormatVersion = __enum_SaverDef_CheckpointFormatVersion()

mutable struct SaverDef <: ProtoType
    filename_tensor_name::AbstractString
    save_tensor_name::AbstractString
    restore_op_name::AbstractString
    max_to_keep::Int32
    sharded::Bool
    keep_checkpoint_every_n_hours::Float32
    version::Int32
    SaverDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SaverDef

export SaverDef_CheckpointFormatVersion, SaverDef
