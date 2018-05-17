using PyCall
using MacroTools

const py_tf = Ref{PyObject}()
const py_tf_core = Ref{PyObject}()
const pywrap_tensorflow = Ref{PyObject}()

function init()
    try
        py_tf[] = pyimport("tensorflow")
        py_tf_core[] = pyimport("tensorflow.core")
        pywrap_tensorflow[] = pyimport("tensorflow.python.pywrap_tensorflow")
    catch err
        error("The Python TensorFlow package could not be imported. You must install Python TensorFlow before using this package.")
    end
end

function py_with(f, ctx_mngr)
    ctx_mngr[:__enter__]()
    f()
    ctx_mngr[:__exit__](nothing, nothing, nothing)
end

function py_bytes(b::Vector{UInt8})
    PyCall.PyObject(ccall(@PyCall.pysym(PyCall.PyString_FromStringAndSize), PyCall.PyPtr, (Ptr{UInt8}, Int), b, sizeof(b)))
end

py_bytes(s::AbstractString) = py_bytes(Vector{UInt8}(s))

macro py_catch(ex)
    target = @match ex begin
        (target_ = value_) => target
    end
    if target !== nothing
        local_block = :(local $(esc(target)))
    else
        local_block = nothing
    end
    quote
        $local_block
        try
            $(esc(ex))
        catch err
            s = string("Python error: ", repr(err.val))
            error(s)
        end
    end
end

function make_py_graph(graph_proto)
    py_graph = py_tf[][:Graph]()
    py_with(py_graph[:as_default]()) do
        # graph_def = py_tf[][:GraphDef]()
        graph_def = py_tf_core[][:protobuf][:meta_graph_pb2][:MetaGraphDef]()
        graph_def[:ParseFromString](graph_proto|>py_bytes)
        # @py_catch py_tf[][:import_graph_def](graph_def, name="")
        @py_catch py_tf[][:train][:import_meta_graph](graph_def)
    end
    py_graph
end

function to_protos(py_graph)
    nodes = PyVector(py_graph[:node])
    n_nodes = length(nodes)
    protos = []
    for node_idx in 1:n_nodes
        node_py = nodes[node_idx]
        proto = Vector{UInt8}(node_py[:SerializeToString]())
        push!(protos, proto)
    end
    return protos
end

function py_gradients(jl_graph_proto, x_names, y_names, grad_y_names)
    py_graph = make_py_graph(jl_graph_proto)

    to_py_node(node_name) = py_graph[:get_tensor_by_name](string(node_name[1], ":", node_name[2]-1))
    to_py_node(node_names::AbstractVector) = tuple(to_py_node.(node_names)...) #Need tuple as Vector will not be accepted
    to_py_node(::Void) = nothing

    py_x = to_py_node(x_names)
    py_y = to_py_node(y_names)
    py_grad_y = to_py_node(grad_y_names)
    @py_catch grad_node = py_tf[][:gradients](py_y, py_x, py_grad_y)
    py_graph_def = py_graph[:as_graph_def]()
    grad_names = []
    for node in grad_node
        if node === nothing
            push!(grad_names, nothing)
            continue
        end
        try
            push!(grad_names, (node[:values][:name], node[:indices][:name]))
        catch
            push!(grad_names, node[:name])
        end
    end
    return to_protos(py_graph_def), grad_names
end
