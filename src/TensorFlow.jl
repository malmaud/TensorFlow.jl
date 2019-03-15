module TensorFlow

@warn("Loading a new version of TensorFlow.jl for the first time. This initial load can take around 5 minutes as code is precompiled; subsequent usage will only take a few seconds.")

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
unsorted_segment_sum,
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
zeros_initializer,
RandomShuffleQueue,
FIFOQueue,
enqueue,
enqueue_many,
dequeue,
dequeue_many,
get_all_op_list,
Ops,
slice,
import_op,
@tfimport,
tf_versioninfo,
copy_to_device,
enable_eager_execution,
EagerTensor,
summary,
create_tape,
set_tape,
with_tape


using Distributed

# Load these packages here so they are available to the additional
# process spawned in 'load_python_process. Arslan thinks that will
# work for now.
using MacroTools  
using PyCall

const pyproc = Ref(0)

function deallocator(data, len, arg)

end

include("context.jl")

function __init__()
    c_deallocator[] = @cfunction(deallocator, Cvoid, (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}))
    for context in default_context()
        push!(global_context, context)
    end
end

function load_python_process(;force_reload=false)
    if myid() == 1
        (pyproc[] > 0 && !force_reload) && return pyproc[] # Python process already loaded
        # Use the TensorFlow.jl Project enviroment
        withenv("JULIA_PROJECT"=>dirname(@__DIR__)) do
            addprocs(1)
        end
        pyproc[] = nprocs()
        py_file = joinpath(dirname(@__FILE__), "py.jl")
        Base.eval(Main, quote
            # These have to be split for unclear reasons on .6
            using Distributed
            remotecall_wait($(pyproc[]), $py_file) do py_file
                include(py_file)
            end
            remotecall_wait($(pyproc[])) do
                init()
            end
        end)
        py_version_check()
        return pyproc[]
    else
        Distributed.remotecall_fetch(1) do
            load_python_process()
        end
    end
end

"""
    @py_proc(block)

Run the given code block in the Julia worker with the Python TensorFlow
library loaded.

Returns a future to the result.
*Warning*: Calling `fetch` on a result that contains a pointer, such as a
`PyObject`, will zero-out the pointer.
"""
macro py_proc(expr)
    quote
        Base.eval(Main, quote
            remotecall_wait($(TensorFlow.load_python_process())) do
                $($(Expr(:quote, expr)))
            end
        end)
    end
end


include("meta.jl")
include("constants.jl")
include("tensorflow_protos.jl")
include("core.jl")
include("eager.jl")
include("run.jl")
include("version.jl")
include("ops.jl")

include("variable.jl")
using .Variables

include("train.jl")
include("io.jl")
include("summary.jl")
include("deprecated.jl")
include("show.jl")
include("generate_ops.jl")
include("tape.jl")
include("keras.jl")

end
