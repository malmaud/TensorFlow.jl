# syntax: proto3
using ProtoBuf
import ProtoBuf.meta
import ._ProtoBuf_Top_.tensorflow

mutable struct CreateSessionRequest <: ProtoType
    graph_def::GraphDef
    config::ConfigProto
    target::AbstractString
    CreateSessionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateSessionRequest

mutable struct CreateSessionResponse <: ProtoType
    session_handle::AbstractString
    graph_version::Int64
    CreateSessionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CreateSessionResponse

mutable struct ExtendSessionRequest <: ProtoType
    session_handle::AbstractString
    graph_def::GraphDef
    current_graph_version::Int64
    ExtendSessionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExtendSessionRequest

mutable struct ExtendSessionResponse <: ProtoType
    new_graph_version::Int64
    ExtendSessionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ExtendSessionResponse
const __fnum_ExtendSessionResponse = Int[4]
meta(t::Type{ExtendSessionResponse}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ExtendSessionResponse, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct RunStepRequest <: ProtoType
    session_handle::AbstractString
    feed::Base.Vector{NamedTensorProto}
    fetch::Base.Vector{AbstractString}
    target::Base.Vector{AbstractString}
    options::RunOptions
    partial_run_handle::AbstractString
    store_errors_in_response_body::Bool
    RunStepRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunStepRequest

mutable struct RunStepResponse <: ProtoType
    tensor::Base.Vector{NamedTensorProto}
    metadata::RunMetadata
    status_code::Int32
    status_error_message::AbstractString
    RunStepResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunStepResponse

mutable struct PartialRunSetupRequest <: ProtoType
    session_handle::AbstractString
    feed::Base.Vector{AbstractString}
    fetch::Base.Vector{AbstractString}
    target::Base.Vector{AbstractString}
    PartialRunSetupRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PartialRunSetupRequest

mutable struct PartialRunSetupResponse <: ProtoType
    partial_run_handle::AbstractString
    PartialRunSetupResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct PartialRunSetupResponse

mutable struct CloseSessionRequest <: ProtoType
    session_handle::AbstractString
    CloseSessionRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CloseSessionRequest

mutable struct CloseSessionResponse <: ProtoType
    CloseSessionResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct CloseSessionResponse

mutable struct ResetRequest <: ProtoType
    container::Base.Vector{AbstractString}
    device_filters::Base.Vector{AbstractString}
    ResetRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ResetRequest

mutable struct ResetResponse <: ProtoType
    ResetResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ResetResponse

mutable struct ListDevicesRequest <: ProtoType
    session_handle::AbstractString
    ListDevicesRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ListDevicesRequest

mutable struct ListDevicesResponse <: ProtoType
    local_device::Base.Vector{DeviceAttributes}
    remote_device::Base.Vector{DeviceAttributes}
    ListDevicesResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ListDevicesResponse

mutable struct MakeCallableRequest <: ProtoType
    session_handle::AbstractString
    options::CallableOptions
    MakeCallableRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MakeCallableRequest

mutable struct MakeCallableResponse <: ProtoType
    handle::Int64
    MakeCallableResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct MakeCallableResponse

mutable struct RunCallableRequest <: ProtoType
    session_handle::AbstractString
    handle::Int64
    feed::Base.Vector{TensorProto}
    RunCallableRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunCallableRequest

mutable struct RunCallableResponse <: ProtoType
    fetch::Base.Vector{TensorProto}
    metadata::RunMetadata
    RunCallableResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct RunCallableResponse

mutable struct ReleaseCallableRequest <: ProtoType
    session_handle::AbstractString
    handle::Int64
    ReleaseCallableRequest(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReleaseCallableRequest

mutable struct ReleaseCallableResponse <: ProtoType
    ReleaseCallableResponse(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReleaseCallableResponse

export CreateSessionRequest, CreateSessionResponse, ExtendSessionRequest, ExtendSessionResponse, RunStepRequest, RunStepResponse, PartialRunSetupRequest, PartialRunSetupResponse, CloseSessionRequest, CloseSessionResponse, ResetRequest, ResetResponse, ListDevicesRequest, ListDevicesResponse, MakeCallableRequest, MakeCallableResponse, RunCallableRequest, RunCallableResponse, ReleaseCallableRequest, ReleaseCallableResponse
