# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct ReaderBaseState <: ProtoType
    work_started::Int64
    work_finished::Int64
    num_records_produced::Int64
    current_work::Array{UInt8,1}
    ReaderBaseState(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReaderBaseState

export ReaderBaseState
