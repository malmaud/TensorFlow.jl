using PyCall
using ProtoBuf
@pyimport tensorflow as py_tf

function py_with(f, ctx_mngr)
    ctx_mngr[:__enter__]()
    f()
    ctx_mngr[:__exit__](nothing, nothing, nothing)
end

function make_py_graph(graph)
    proto = get_proto(graph)
    py_graph = py_tf.Graph()
    py_with(py_graph[:as_default]()) do
        graph_def = py_tf.GraphDef()
        graph_def[:ParseFromString](proto|>String)
        py_tf.import_graph_def(graph_def, name="")
    end
    py_graph
end

function extend_graph(graph, other::PyObject)
    n_nodes = length(other[:node])
    nodes = []
    for node_idx in 1:n_nodes
        node_py = other[:node][node_idx]
        proto = node_py[:SerializeToString]().data
        b = IOBuffer()
        write(b, proto)
        seekstart(b)
        node_def = tensorflow.NodeDef()
        readproto(b, node_def)
        if isnull(get_node_by_name(graph, node_def.name))
            push!(nodes, node_def)
        end
    end
    for node in nodes
        Node(node)
    end
end


function gradients(y, x::AbstractArray)
    py_graph = make_py_graph(get_def_graph())
    to_py_node = node->py_graph[:get_tensor_by_name](string(node_name(node), ":0"))
    py_x = [to_py_node(node) for node in x]
    py_y = to_py_node(y)
    grad_node = py_tf.gradients(py_y, py_x)
    py_graph_def = py_graph[:as_graph_def]()
    extend_graph(get_def_graph(), py_graph_def)
    return [get_node_by_name(get_def_graph(), _[:name])|>get for _ in grad_node]
end

gradients(y, x) = gradients(y, [x])[1]
