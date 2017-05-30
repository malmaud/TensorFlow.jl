using Base.Test
using TensorFlow

@testset "TensorShape" begin
    @test TensorShape([]) == TensorShape(Nullable{Int}[],false)
    @test TensorShape(collect(tuple())) == TensorShape([],false)
    
    @test TensorShape([-1, 15]) == TensorShape([Nullable{Int}(), Nullable{Int}(15)], false)
    @test TensorShape([10, 12]) == TensorShape([Nullable{Int}(10), Nullable{Int}(12)], false)

    @test TensorShape(nothing).rank_unknown == true
end

@testset "Graph importing" begin
    if tf_version() >= v"1.0.0-rc1"
        graph_pb = read(joinpath(dirname(@__FILE__), "graph.pb"))
        graph = Graph()
        sess = Session(graph)
        x_new = constant(Int32(5))
        options = GraphImportOptions()
        options.input_mapping[("x", 1)] = x_new
        push!(options.return_output, ("z", 1))
        z = import_graph_def(graph, graph_pb, options)
        @test run(sess, z) == [Int32(7)]
    end
end

@testset "get_operations" begin
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
end


@testset "Graph Node Access By Name" begin
    g = Graph()
    sess= Session(g)
    x = placeholder(Float64, name="x")
    @test g["x"] == x
end

@testset "Disconnected gradients" begin
    let
        as_default(Graph()) do
            unused = get_variable("unused", [], Float64)
            used = get_variable("used", [], Float64)
            loss = used.^2
            optimizer = train.minimize(train.AdamOptimizer(), loss)
            # This would have thrown an error if Disconnected gradients were causing issues
        end
    end
end


@testset "Gradients" begin
    let
        sess = Session(Graph())
        A = get_variable("A", (1,), Float32)
        B = get_variable("B", (1,), Float32)

        @test [[2.0f0]] == run(sess, gradients(2A, [A]))
        @test [2.0f0] == run(sess, gradients(2A, A))

        @test [[3.0f0], [5.0f0]] == run(sess, gradients(3A+5B, [A, B]))
        @test [[8.0f0]] == run(sess, gradients([3A, 5A], [A]))

        @test [[9.0f0], [3.0f0]] == run(sess, gradients([2A+3B, 7A], [A, B]))

        @test [35.0f0] == run(sess, gradients(7A, A, constant([5.0f0])))
        @test [68.0f0] == run(sess, gradients([7A,3A], A, [constant([5.0f0]), constant([11.0f0])]))
        @test [38.0f0] == run(sess, gradients([7A,3A], A, [constant([5.0f0]), nothing]))

    end

end

