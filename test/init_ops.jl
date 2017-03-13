using TensorFlow
using Base.Test

sess = Session(Graph())

c = ConstantInitializer(2.0)
@test rand(c, 2, 3) == fill(2.0, 2, 3)

let
    initializer = zeros_initializer()
    @test run(sess, initializer([2,3])) == zeros(2,3)
end

let
    initializer = ones_initializer(Float64)
    @test run(sess, initializer([2,3])) == ones(2, 3)
end
