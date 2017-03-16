using Base.Test
using TensorFlow


@testset "Naming" begin
    let
        g = Graph()
        local i, j_jl, j, k, ijk, ij, fq, m, W, Y
        as_default(g) do
            @tf begin
                i = constant(1.0)
                j_jl=rand() # some nonTF function, this would error if name incorrectly inserted
                j = constant(j_jl)
                k = get_variable([], Float64)
                m = get_variable("namefor_m", [], Float64)

                ijk = add_n([i,j,k])
                ij = add_n([i,j]; name="namefor_ij")
                fq = TensorFlow.FIFOQueue(10, [Int32, Int64]);

                X = get_variable([50], Float64)
                variable_scope("logisitic_model") do
                     W = get_variable([50, 10], Float64)
                     B = get_variable([10], Float64)
                     Y = nn.softmax(X * W + B)
                 end

            end
        end

        @test i == get_tensor_by_name(g, "i")
        @test j == get_tensor_by_name(g, "j")
        dump(get_tensor_by_name(g, "k"))
        @test Tensor(k.var_node) == get_tensor_by_name(g, "k")
        @test Tensor(k.assign_node) == get_tensor_by_name(g, "k/Assign")
        @test ijk == get_tensor_by_name(g, "ijk")
        @test fq.op.ptr == get_tensor_by_name(g, "fq").op.ptr
        @test ij == get_tensor_by_name(g, "namefor_ij")
        @test Tensor(m.var_node) == get_tensor_by_name(g, "namefor_m")

        @test Y == get_tensor_by_name(g, "Y")
        @test Tensor(W.var_node) == get_tensor_by_name(g, "logisitic_model/W")
        @test Tensor(W.assign_node) == get_tensor_by_name(g, "logisitic_model/W/Assign")
    end
end

@testset "While" begin
    let
        sess = Session(Graph())
        i = constant(1)
        loop_sum = constant(0)
        res = @tf while i ≤ 10
            sq = i.*i
            [i=>i+1, loop_sum=>loop_sum+sq]
        end
        @test run(sess, res) == [11, 385]
    end
end
