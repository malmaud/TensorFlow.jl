# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

mutable struct TensorProto <: ProtoType
    dtype::Int32
    tensor_shape::TensorShapeProto
    version_number::Int32
    tensor_content::Array{UInt8,1}
    half_val::Base.Vector{Int32}
    float_val::Base.Vector{Float32}
    double_val::Base.Vector{Float64}
    int_val::Base.Vector{Int32}
    string_val::Base.Vector{Array{UInt8,1}}
    scomplex_val::Base.Vector{Float32}
    int64_val::Base.Vector{Int64}
    bool_val::Base.Vector{Bool}
    dcomplex_val::Base.Vector{Float64}
    resource_handle_val::Base.Vector{ResourceHandleProto}
    variant_val::Base.Any
    uint32_val::Base.Vector{UInt32}
    uint64_val::Base.Vector{UInt64}
    TensorProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorProto (has cyclic type dependency)
const __fnum_TensorProto = Int[1,2,3,4,13,5,6,7,8,9,10,11,12,14,15,16,17]
const __pack_TensorProto = Symbol[:half_val,:float_val,:double_val,:int_val,:scomplex_val,:int64_val,:bool_val,:dcomplex_val,:uint32_val,:uint64_val]
const __ftype_TensorProto = Dict(:variant_val => "Base.Vector{VariantTensorDataProto}")
meta(t::Type{TensorProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorProto, ProtoBuf.DEF_VAL, true, __pack_TensorProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_TensorProto)

mutable struct VariantTensorDataProto <: ProtoType
    type_name::AbstractString
    metadata::Array{UInt8,1}
    tensors::Base.Vector{TensorProto}
    VariantTensorDataProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct VariantTensorDataProto (has cyclic type dependency)

export TensorProto, VariantTensorDataProto, TensorProto, VariantTensorDataProto
