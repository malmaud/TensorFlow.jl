using TensorFlow
using Test


@testset "Importing" begin
    # Test importing an operation not in the default list
    session = Session(Graph())
    as_string = import_op("AsString")
    @test run(session, as_string(2)) == "2"
end



import_op("Atan2") #Must be outside testset as per https://github.com/JuliaLang/julia/issues/27244
@testset "Importing a name that is used by Base" begin
    session = Session(Graph())
    @test run(session, Ops.atan2(Tensor(1.0), Tensor(1.0))) > 0 
end
