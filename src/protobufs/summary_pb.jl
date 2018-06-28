# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

mutable struct HistogramProto
    min::Float64
    max::Float64
    num::Float64
    sum::Float64
    sum_squares::Float64
    bucket_limit::Array{Float64,1}
    bucket::Array{Float64,1}
    HistogramProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type HistogramProto
const __pack_HistogramProto = Symbol[:bucket_limit,:bucket]
meta(t::Type{HistogramProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_HistogramProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES)
hash(v::HistogramProto) = ProtoBuf.protohash(v)
isequal(v1::HistogramProto, v2::HistogramProto) = ProtoBuf.protoisequal(v1, v2)
==(v1::HistogramProto, v2::HistogramProto) = ProtoBuf.protoeq(v1, v2)

mutable struct Summary_Image
    height::Int32
    width::Int32
    colorspace::Int32
    encoded_image_string::Array{UInt8,1}
    Summary_Image(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type Summary_Image
hash(v::Summary_Image) = ProtoBuf.protohash(v)
isequal(v1::Summary_Image, v2::Summary_Image) = ProtoBuf.protoisequal(v1, v2)
==(v1::Summary_Image, v2::Summary_Image) = ProtoBuf.protoeq(v1, v2)

mutable struct Summary_Audio
    sample_rate::Float32
    num_channels::Int64
    length_frames::Int64
    encoded_audio_string::Array{UInt8,1}
    content_type::AbstractString
    Summary_Audio(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type Summary_Audio
hash(v::Summary_Audio) = ProtoBuf.protohash(v)
isequal(v1::Summary_Audio, v2::Summary_Audio) = ProtoBuf.protoisequal(v1, v2)
==(v1::Summary_Audio, v2::Summary_Audio) = ProtoBuf.protoeq(v1, v2)

mutable struct Summary_Value
    node_name::AbstractString
    tag::AbstractString
    simple_value::Float32
    obsolete_old_style_histogram::Array{UInt8,1}
    image::Summary_Image
    histo::HistogramProto
    audio::Summary_Audio
    tensor::TensorProto
    Summary_Value(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type Summary_Value
const __fnum_Summary_Value = Int[7,1,2,3,4,5,6,8]
const __oneofs_Summary_Value = Int[0,0,1,1,1,1,1,1]
const __oneof_names_Summary_Value = [Symbol("value")]
meta(t::Type{Summary_Value}) = meta(t, ProtoBuf.DEF_REQ, __fnum_Summary_Value, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_Summary_Value, __oneof_names_Summary_Value)
hash(v::Summary_Value) = ProtoBuf.protohash(v)
isequal(v1::Summary_Value, v2::Summary_Value) = ProtoBuf.protoisequal(v1, v2)
==(v1::Summary_Value, v2::Summary_Value) = ProtoBuf.protoeq(v1, v2)

mutable struct Summary
    value::Array{Summary_Value,1}
    Summary(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type Summary
hash(v::Summary) = ProtoBuf.protohash(v)
isequal(v1::Summary, v2::Summary) = ProtoBuf.protoisequal(v1, v2)
==(v1::Summary, v2::Summary) = ProtoBuf.protoeq(v1, v2)

export HistogramProto, Summary_Image, Summary_Audio, Summary_Value, Summary
