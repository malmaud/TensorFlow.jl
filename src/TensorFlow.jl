module TensorFlow

export
Graph,
get_collection,
get_def_graph,
Session,
Node,
AbstractNode,
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
reduce_mean

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
