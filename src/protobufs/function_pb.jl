# syntax: proto3
using Compat
using ProtoBuf
import ProtoBuf.meta
import Base: hash, isequal, ==

type FunctionDef_Node_AttrEntry
    key::AbstractString
    value::AttrValue
    FunctionDef_Node_AttrEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FunctionDef_Node_AttrEntry (mapentry)
hash(v::FunctionDef_Node_AttrEntry) = ProtoBuf.protohash(v)
isequal(v1::FunctionDef_Node_AttrEntry, v2::FunctionDef_Node_AttrEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::FunctionDef_Node_AttrEntry, v2::FunctionDef_Node_AttrEntry) = ProtoBuf.protoeq(v1, v2)

type FunctionDef_Node
    ret::Array{AbstractString,1}
    op::AbstractString
    arg::Array{AbstractString,1}
    dep::Array{AbstractString,1}
    attr::Dict{AbstractString,AttrValue} # map entry
    FunctionDef_Node(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FunctionDef_Node
hash(v::FunctionDef_Node) = ProtoBuf.protohash(v)
isequal(v1::FunctionDef_Node, v2::FunctionDef_Node) = ProtoBuf.protoisequal(v1, v2)
==(v1::FunctionDef_Node, v2::FunctionDef_Node) = ProtoBuf.protoeq(v1, v2)

type FunctionDef_RetEntry
    key::AbstractString
    value::AbstractString
    FunctionDef_RetEntry(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FunctionDef_RetEntry (mapentry)
hash(v::FunctionDef_RetEntry) = ProtoBuf.protohash(v)
isequal(v1::FunctionDef_RetEntry, v2::FunctionDef_RetEntry) = ProtoBuf.protoisequal(v1, v2)
==(v1::FunctionDef_RetEntry, v2::FunctionDef_RetEntry) = ProtoBuf.protoeq(v1, v2)

type FunctionDef
    signature::OpDef
    node::Array{FunctionDef_Node,1}
    node_def::Array{NodeDef,1}
    ret::Dict{AbstractString,AbstractString} # map entry
    FunctionDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FunctionDef
hash(v::FunctionDef) = ProtoBuf.protohash(v)
isequal(v1::FunctionDef, v2::FunctionDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::FunctionDef, v2::FunctionDef) = ProtoBuf.protoeq(v1, v2)

type GradientDef
    function_name::AbstractString
    gradient_func::AbstractString
    GradientDef(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type GradientDef
hash(v::GradientDef) = ProtoBuf.protohash(v)
isequal(v1::GradientDef, v2::GradientDef) = ProtoBuf.protoisequal(v1, v2)
==(v1::GradientDef, v2::GradientDef) = ProtoBuf.protoeq(v1, v2)

type FunctionDefLibrary
    _function::Array{FunctionDef,1}
    gradient::Array{GradientDef,1}
    FunctionDefLibrary(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #type FunctionDefLibrary
hash(v::FunctionDefLibrary) = ProtoBuf.protohash(v)
isequal(v1::FunctionDefLibrary, v2::FunctionDefLibrary) = ProtoBuf.protoisequal(v1, v2)
==(v1::FunctionDefLibrary, v2::FunctionDefLibrary) = ProtoBuf.protoeq(v1, v2)

export FunctionDefLibrary, FunctionDef_Node_AttrEntry, FunctionDef_Node, FunctionDef_RetEntry, FunctionDef, GradientDef
# mapentries: Pair{AbstractString,Tuple{AbstractString,AbstractString}}("FunctionDef_RetEntry",("AbstractString","AbstractString")), Pair{AbstractString,Tuple{AbstractString,AbstractString}}("FunctionDef_Node_AttrEntry",("AbstractString","AttrValue"))
