# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta

mutable struct SummaryDescription <: ProtoType
    type_hint::AbstractString
    SummaryDescription(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SummaryDescription

mutable struct HistogramProto <: ProtoType
    min::Float64
    max::Float64
    num::Float64
    sum::Float64
    sum_squares::Float64
    bucket_limit::Base.Vector{Float64}
    bucket::Base.Vector{Float64}
    HistogramProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct HistogramProto
const __pack_HistogramProto = Symbol[:bucket_limit,:bucket]
meta(t::Type{HistogramProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_HistogramProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct SummaryMetadata_PluginData <: ProtoType
    plugin_name::AbstractString
    content::Array{UInt8,1}
    SummaryMetadata_PluginData(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SummaryMetadata_PluginData

mutable struct SummaryMetadata <: ProtoType
    plugin_data::SummaryMetadata_PluginData
    display_name::AbstractString
    summary_description::AbstractString
    SummaryMetadata(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SummaryMetadata

mutable struct Summary_Image <: ProtoType
    height::Int32
    width::Int32
    colorspace::Int32
    encoded_image_string::Array{UInt8,1}
    Summary_Image(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Summary_Image

mutable struct Summary_Audio <: ProtoType
    sample_rate::Float32
    num_channels::Int64
    length_frames::Int64
    encoded_audio_string::Array{UInt8,1}
    content_type::AbstractString
    Summary_Audio(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Summary_Audio

mutable struct Summary_Value <: ProtoType
    node_name::AbstractString
    tag::AbstractString
    metadata::SummaryMetadata
    simple_value::Float32
    obsolete_old_style_histogram::Array{UInt8,1}
    image::Summary_Image
    histo::HistogramProto
    audio::Summary_Audio
    tensor::TensorProto
    Summary_Value(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Summary_Value
const __fnum_Summary_Value = Int[7,1,9,2,3,4,5,6,8]
const __oneofs_Summary_Value = Int[0,0,0,1,1,1,1,1,1]
const __oneof_names_Summary_Value = [Symbol("value")]
meta(t::Type{Summary_Value}) = meta(t, ProtoBuf.DEF_REQ, __fnum_Summary_Value, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_Summary_Value, __oneof_names_Summary_Value, ProtoBuf.DEF_FIELD_TYPES)

mutable struct Summary <: ProtoType
    value::Base.Vector{Summary_Value}
    Summary(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct Summary

export SummaryDescription, HistogramProto, SummaryMetadata_PluginData, SummaryMetadata, Summary_Image, Summary_Audio, Summary_Value, Summary
