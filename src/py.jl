using PyCall

const py_tf = Ref{PyObject}()
const pywrap_tensorflow = Ref{PyObject}()

function py_with(f, ctx_mngr)
    ctx_mngr[:__enter__]()
    f()
    ctx_mngr[:__exit__](nothing, nothing, nothing)
end

function py_bytes(b::Vector{UInt8})
    PyCall.PyObject(ccall(@PyCall.pysym(PyString_FromStringAndSize), PyCall.PyPtr, (Ptr{UInt8}, Int), b, sizeof(b)))
end

function make_py_graph(graph_proto)
    py_graph = py_tf[][:Graph]()
    py_with(py_graph[:as_default]()) do
        graph_def = py_tf[][:GraphDef]()
        graph_def[:ParseFromString](graph_proto|>py_bytes)
        py_tf[][:import_graph_def](graph_def, name="")
    end
    py_graph
end

function to_protos(py_graph)
    n_nodes = length(py_graph[:node])
    protos = []
    for node_idx in 1:n_nodes
        node_py = py_graph[:node][node_idx]
        proto = node_py[:SerializeToString]().data
        push!(protos, proto)
    end
    return protos
end

function py_gradients(jl_graph_proto, x_names, y_name)
    py_graph = make_py_graph(jl_graph_proto)
    to_py_node = node_name->py_graph[:get_tensor_by_name](string(node_name[1], ":", node_name[2]-1))
    py_x = [to_py_node(node) for node in x_names]
    py_y = to_py_node(y_name)
    grad_node = py_tf[][:gradients](py_y, py_x)    
    py_graph_def = py_graph[:as_graph_def]()
    grad_names = [_[:name] for _ in grad_node]
    return to_protos(py_graph_def), grad_names
end

const events_writer = Ref{PyObject}()

function open_events_file(path)
    events_writer[] = pywrap_tensorflow[][:EventsWriter](path)
end

function write_event(event_proto)
    event = py_tf[][:Event]()
    event[:ParseFromString](py_bytes(event_proto))
    writer = events_writer[]
    writer[:WriteEvent](event)
    writer[:Flush]()
end

function close_events_file()
    writer = events_writer[]
    writer[:Close]()
end
