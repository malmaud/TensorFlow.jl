# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

struct __enum_Code <: ProtoEnum
    OK::Int32
    CANCELLED::Int32
    UNKNOWN::Int32
    INVALID_ARGUMENT::Int32
    DEADLINE_EXCEEDED::Int32
    NOT_FOUND::Int32
    ALREADY_EXISTS::Int32
    PERMISSION_DENIED::Int32
    UNAUTHENTICATED::Int32
    RESOURCE_EXHAUSTED::Int32
    FAILED_PRECONDITION::Int32
    ABORTED::Int32
    OUT_OF_RANGE::Int32
    UNIMPLEMENTED::Int32
    INTERNAL::Int32
    UNAVAILABLE::Int32
    DATA_LOSS::Int32
    DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_::Int32
    __enum_Code() = new(0,1,2,3,4,5,6,7,16,8,9,10,11,12,13,14,15,20)
end #struct __enum_Code
const Code = __enum_Code()

export Code
