using Base.Test
using TensorFlow

#################
# Graph importing
#################

if tf_version() >= v"1.0.0-rc1"
    graph_pb = read(joinpath(dirname(@__FILE__), "graph.pb"))
    graph = Graph()
    sess = Session(graph)
    x_new = constant(Int32(5))
    options = GraphInputOptions()
    options.input_mapping[("x", 1)] = x_new
    push!(options.return_output, ("z", 1))
    z = import_graph_def(graph, graph_pb, options)
    @test run(sess, z) == [Int32(7)]
end

#################
# get_operations
#################
let
    graph = Graph()
    TensorFlow.set_def_graph(graph)
    x = placeholder(Int32, name="x")
    y = placeholder(Int32, name="y")
    z = TensorFlow.add(x, y, name="z")
    names = Set{String}()
    for op in get_operations(graph)
        push!(names, get_def(op).name)
    end
    @test length(names) == 3
    for name in ["x", "y", "z"]
        @test name ∈ names
    end
end

let
    graph = Graph()
    local x
    as_default(graph) do
        x = placeholder(Int32, name="x")
    end
    x_retrieved = get_tensor_by_name(graph, "x:0")
    @test x == x_retrieved
end

let
    graph = Graph()
    sess = Session(graph)
    local x
    with_device("cpu:1") do
        x = constant(1)
    end
    @test run(sess, x) == 1
end
