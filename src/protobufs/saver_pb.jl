# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type __enum_SaverDef_CheckpointFormatVersion <: ProtoEnum
    LEGACY::Int32
    V1::Int32
    V2::Int32
    __enum_SaverDef_CheckpointFormatVersion() = new(0,1,2)
end #type __enum_SaverDef_CheckpointFormatVersion
const SaverDef_CheckpointFormatVersion = __enum_SaverDef_CheckpointFormatVersion()

type SaverDef
    filename_tensor_name::AbstractString
    save_tensor_name::AbstractString
    restore_op_name::AbstractString
    max_to_keep::Int32
    sharded::Bool
    keep_checkpoint_every_n_hours::Float32
    version::Int32
    SaverDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type SaverDef
hash(v::SaverDef) = ProtoBuf.protohash(v)
isequal(v1::SaverDef, v2::SaverDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::SaverDef, v2::SaverDef) = ProtoBuf.protoeq(v1, v2)

export SaverDef_CheckpointFormatVersion, SaverDef
