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
image_summary

function __init__()
    if myid() == 1
        set_def_graph(Graph())
        spawn_py_process()
    end
end

include("constants.jl")
include("tensorflow_protos.jl")
include("core.jl")
include("variable.jl")
include("ops.jl")
include("train.jl")


end
