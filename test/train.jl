using TensorFlow
using Base.Test

@testset "train.save and train.resore" begin
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


@testset "metagraph importing and exporting" begin
    mktempdir() do tmppath
        modelfile = joinpath(tmppath, "my_model")
        let
            session = Session(Graph())
            x = constant(1)
            @tf y = x+1
            train.export_meta_graph(modelfile)
        end

        let
            session = Session(Graph())
            train.import_meta_graph(modelfile)
            y = get_tensor_by_name("y")
            @test run(session, y) == 2
        end
    end
end
