# Hack until ProtoBuf.jl is fixed
module google
end

__precompile__(true)
module TensorFlow

export
Graph,
get_collection,
get_def_graph,
Session,
Operation,
AbstractOperation,
get_graph,
node_name,
run,
get_proto,
get_node_by_name,
get_shape,
get_def,
Operation,
gradients,
placeholder,
constant,
concat,
cast,
read_file,
pack,
expand_dims,
argmin,
one_hot,
random_uniform,
random_normal,
nn,
image,
Variable,
assign,
initialize_all_variables,
variable_scope,
get_variable,
ConstantInitializer,
train,
reduce_sum,
reduce_prod,
reduce_min,
reduce_max,
reduce_all,
reduce_any,
reduce_mean,
equal,
not_equal,
less_equal,
greater,
greater_equal,
logical_and,
logical_not,
logical_or,
logical_xor,
strided_slice,
unpack,
tile,
pad,
gather,
gather_nd,
dynamic_partition,
dynamic_stitch,
boolean_mask,
where,
scalar_summary,
histogram_summary,
merge_summary,
merge_all_summaries,
image_summary,
io,
AbstractTensor,
Tensor,
add_n,
clip_by_value,
clip_by_norm,
clip_by_average_norm,
clip_by_global_norm,
global_norm


function __init__()
    c_deallocator[] = cfunction(deallocator, Void, (Ptr{Void}, Csize_t, Ptr{Void}))
    set_def_graph(Graph())
    try
        py_tf[] = pyimport("tensorflow")
        pywrap_tensorflow[] = pyimport("tensorflow.python.pywrap_tensorflow")
    catch err
        error("The Python TensorFlow package could not be imported. You must install Python TensorFlow before using this package.")
    end
end

abstract AbstractTensorShape
include("constants.jl")
include("tensorflow_protos.jl")
include("core.jl")
include("variable.jl")
include("shape_inference.jl")
using .ShapeInference
export get_shape


include("ops.jl")
include("train.jl")
include("io.jl")

end
