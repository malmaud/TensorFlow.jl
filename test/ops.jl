using TensorFlow
using Base.Test


@testset "Importing" begin
    # Test importing an operation not in the default list
    session = Session(Graph())
    as_string = import_op("AsString")
    @test run(session, as_string(2)) == "2"
end
