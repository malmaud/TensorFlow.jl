# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

struct __enum__DataType <: ProtoEnum
    DT_INVALID::Int32
    DT_FLOAT::Int32
    DT_DOUBLE::Int32
    DT_INT32::Int32
    DT_UINT8::Int32
    DT_INT16::Int32
    DT_INT8::Int32
    DT_STRING::Int32
    DT_COMPLEX64::Int32
    DT_INT64::Int32
    DT_BOOL::Int32
    DT_QINT8::Int32
    DT_QUINT8::Int32
    DT_QINT32::Int32
    DT_BFLOAT16::Int32
    DT_QINT16::Int32
    DT_QUINT16::Int32
    DT_UINT16::Int32
    DT_COMPLEX128::Int32
    DT_HALF::Int32
    DT_RESOURCE::Int32
    DT_VARIANT::Int32
    DT_UINT32::Int32
    DT_UINT64::Int32
    DT_FLOAT_REF::Int32
    DT_DOUBLE_REF::Int32
    DT_INT32_REF::Int32
    DT_UINT8_REF::Int32
    DT_INT16_REF::Int32
    DT_INT8_REF::Int32
    DT_STRING_REF::Int32
    DT_COMPLEX64_REF::Int32
    DT_INT64_REF::Int32
    DT_BOOL_REF::Int32
    DT_QINT8_REF::Int32
    DT_QUINT8_REF::Int32
    DT_QINT32_REF::Int32
    DT_BFLOAT16_REF::Int32
    DT_QINT16_REF::Int32
    DT_QUINT16_REF::Int32
    DT_UINT16_REF::Int32
    DT_COMPLEX128_REF::Int32
    DT_HALF_REF::Int32
    DT_RESOURCE_REF::Int32
    DT_VARIANT_REF::Int32
    DT_UINT32_REF::Int32
    DT_UINT64_REF::Int32
    __enum__DataType() = new(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123)
end #struct __enum__DataType
const _DataType = __enum__DataType()

export _DataType
