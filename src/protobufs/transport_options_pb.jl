# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct RecvBufRespExtra <: ProtoType
    tensor_content::Base.Vector{Array{UInt8,1}}
    RecvBufRespExtra(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RecvBufRespExtra

export RecvBufRespExtra
