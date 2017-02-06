using Base.Test

k = placeholder(Float32; shape=[10,20, -1])
@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 3)
@test_throws BoundsError get_shape(k, 4)

@test_throws ErrorException get_shape(placeholder(Float32), 1)

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
