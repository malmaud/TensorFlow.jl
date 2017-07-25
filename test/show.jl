using TensorFlow
using Base.Test

@testset "Tensorboard" begin
    @test_nowarn TensorFlow.get_tensorboard()
end
