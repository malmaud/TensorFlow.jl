# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct NewReplaySession <: ProtoType
    devices::ListDevicesResponse
    session_handle::AbstractString
    NewReplaySession(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NewReplaySession

mutable struct ReplayOp <: ProtoType
    start_time_us::Float64
    end_time_us::Float64
    create_session::CreateSessionRequest
    extend_session::ExtendSessionRequest
    partial_run_setup::PartialRunSetupRequest
    run_step::RunStepRequest
    close_session::CloseSessionRequest
    list_devices::ListDevicesRequest
    reset_request::ResetRequest
    make_callable::MakeCallableRequest
    run_callable::RunCallableRequest
    release_callable::ReleaseCallableRequest
    new_replay_session::NewReplaySession
    create_session_response::CreateSessionResponse
    extend_session_response::ExtendSessionResponse
    partial_run_setup_response::PartialRunSetupResponse
    run_step_response::RunStepResponse
    close_session_response::CloseSessionResponse
    list_devices_response::ListDevicesResponse
    reset_request_response::ResetResponse
    make_callable_response::MakeCallableResponse
    run_callable_response::RunCallableResponse
    release_callable_response::ReleaseCallableResponse
    ReplayOp(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ReplayOp
const __fnum_ReplayOp = Int[31,32,1,2,3,4,5,6,7,8,9,10,11,21,22,23,24,25,26,27,28,29,30]
const __oneofs_ReplayOp = Int[0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
const __oneof_names_ReplayOp = [Symbol("op"),Symbol("response")]
meta(t::Type{ReplayOp}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ReplayOp, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_ReplayOp, __oneof_names_ReplayOp, ProtoBuf.DEF_FIELD_TYPES)

export NewReplaySession, ReplayOp
