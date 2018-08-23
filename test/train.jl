using TensorFlow
using Test

@testset "save and resore" begin
    try
        let
            session = Session(Graph())
            x = get_variable("x", [], Float32)
            run(session, assign(x, 5.f0))
            saver = train.Saver()
            train.save(saver, session, "weights.jld")
        end

        let
            session = Session(Graph())
            @tf x = get_variable([], Float32)
            saver = train.Saver()
            train.restore(saver, session, "weights.jld")
            @test run(session, x) == 5.0f0
        end
   finally
        rm("weights.jld"; force=true)
   end
end


@testset "save and restore max_to_keep" begin
    try
        let
            session = Session(Graph())
            x = get_variable("x", [], Float32)
            run(session, assign(x, 5.f0))
            saver = train.Saver(max_to_keep=5)
            for i in 1:12
                train.save(saver, session, "weights.jld", global_step=i)
            end
        end

        let
            session = Session(Graph())
            @tf x = get_variable([], Float32)
            saver = train.Saver()
            for i in 1:7
                @test_throws SystemError train.restore(saver, session, "weights.jld-$i")
            end
            for i in 8:12
                train.restore(saver, session, "weights.jld-$i")
                @test run(session, x) == 5.0f0
            end
        end
    finally
        for i in 1:12
            rm("weights.jld-$i"; force=true)
        end
    end
end


@testset "metagraph importing and exporting" begin
    mktempdir() do tmppath
        modelfile = joinpath(tmppath, "my_model")
        let
            session = Session(Graph())
            x = constant(1)
            @tf y = x+1
            z = Variable(3, name="z")
            train.export_meta_graph(modelfile)
        end

        let
            session = Session(Graph())
            train.import_meta_graph(modelfile)
            y = get_tensor_by_name("y")
            @test run(session, y) == 2
            run(session, global_variables_initializer())
            z = get_tensor_by_name("z")
            @test run(session, z) == 3
        end
    end
end
