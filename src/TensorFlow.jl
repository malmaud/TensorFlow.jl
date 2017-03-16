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
stack,
expand_dims,
argmin,
argmax,
one_hot,
random_uniform,
random_normal,
nn,
sign,
image,
Variable,
assign,
assign_add,
assign_sub,
scatter_update,
scatter_sub,
scatter_add,
scatter_mul,
scatter_div,
scatter_nd,
global_variables_initializer,
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
segment_sum,
segment_prod,
segment_min,
segment_max,
segment_mean,
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
unstack,
tile,
pad,
gather,
gather_nd,
dynamic_partition,
dynamic_stitch,
boolean_mask,
where,
is_inf,
is_finite,
is_nan,
io,
AbstractTensor,
Tensor,
add_n,
clip_by_value,
clip_by_norm,
clip_by_average_norm,
clip_by_global_norm,
global_norm,
Print,
import_graph_def,
tf_version,
GraphImportOptions,
get_operations,
while_loop,
get_tensor_by_name,
as_default,
@tf,
withname,
tf_while,
visualize,
visualize_graph,
with_device,
@device_str,
TensorShape,
get_shape,
batch_matmul,
squared_difference,
multiply,
subtract,
divide,
ones_initializer,
zeros_initializer


isdefined(Base, :⊻) || (export ⊻)
isdefined(Base, :slice) || (export slice)

const pyproc = Ref(0)

function __init__()
    c_deallocator[] = cfunction(deallocator, Void, (Ptr{Void}, Csize_t, Ptr{Void}))
end

function load_python_process()
    if myid() == 1
        pyproc[] > 0 && return pyproc[] # Python process already loaded
        addprocs(1)
        pyproc[] = nprocs()
        py_file = joinpath(dirname(@__FILE__), "py.jl")
        eval(Main, quote
            # These have to be split for unclear reasons on .6
            remotecall_wait($(pyproc[]), $py_file) do py_file
                include(py_file)
            end
            remotecall_wait($(pyproc[])) do
                init()
            end
        end)
        return pyproc[]
    else
        remotecall_fetch(1) do
            load_python_process()
        end
    end
end

"""
    @py_proc(block)

Run the given code block in the Julia worker with the Python TensorFlow
library loaded.
"""
macro py_proc(expr)
    quote
        eval(Main, quote
            remotecall_fetch($(TensorFlow.load_python_process())) do
                $($(Expr(:quote, expr)))
            end
        end)
    end
end


include("constants.jl")
include("tensorflow_protos.jl")
include("core.jl")
include("run.jl")
include("variable.jl")
include("shape_inference.jl")
include("meta.jl")
include("ops.jl")
include("train.jl")
include("io.jl")
include("show.jl")
include("summary.jl")
include("deprecated.jl")

end
