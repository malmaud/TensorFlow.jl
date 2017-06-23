using Base.Test
using TensorFlow
using Distributions

@testset "Registering Ops" begin

    @test TensorFlow.is_registered_op(TensorFlow.FIFOQueue) == TensorFlow.RegisteredOp()
    @test TensorFlow.is_registered_op(nn.rnn_cell.GRUCell) == TensorFlow.NotRegisteredOp()

    @test TensorFlow.is_registered_op(typeof(add_n)) == TensorFlow.RegisteredOp()
    @test TensorFlow.is_registered_op(typeof(nn.softmax)) == TensorFlow.RegisteredOp()

    @test_throws MethodError TensorFlow.is_registered_op(add_n)
    @test_throws MethodError TensorFlow.is_registered_op(nn.softmax)
end


@testset "Naming" begin
    let
        g = Graph()
        local i, j_jl, j, k, ijk, ij, ij2, fq, m, W, Y, Ysum1, Ysum2, Ysum3, Ysum4
        as_default(g) do
            @tf begin
                i = constant(1.0)
                j_jl=rand() # some nonTF function, this would error if name incorrectly inserted
                j = constant(j_jl)
                k = get_variable([], Float64)
                m = get_variable("namefor_m", [], Float64)

                ijk = add_n([i,j,k])
                ij = add_n([i,j]; name="namefor_ij")
                ij2 = add_n([i,j], name="namefor_ij2") #comma instead of semicolon

                fq = TensorFlow.FIFOQueue(10, [Int32, Int64]); #This datatype should be an Op
                cc = nn.rnn_cell.GRUCell(40) #this Datatype should not be an Op

                X = get_variable([1, 50], Float64)
                variable_scope("logisitic_model") do
                     W = get_variable([50, 10], Float64)
                     B = get_variable([10], Float64)
                     Y = nn.softmax(X * W + B)
                 end

                Ysum1 = reduce_sum(Y)
                Ysum2 = reduce_sum(Y; keep_dims=true) # With a semicolon
                Ysum3 = reduce_sum(Y, keep_dims=true) # With a comma (issue #188)

                Ysum4 = reduce_sum(Y, keep_dims=true, name="namefor_Ysum4") # With a comma (issue #188)
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
        @test ij2 == get_tensor_by_name(g, "namefor_ij2")
        @test Tensor(m.var_node) == get_tensor_by_name(g, "namefor_m")

        @test Tensor(W.var_node) == get_tensor_by_name(g, "logisitic_model/W")
        @test Tensor(W.assign_node) == get_tensor_by_name(g, "logisitic_model/W/Assign")
        @test Y == get_tensor_by_name(g, "Y")

        @test Ysum1 == get_tensor_by_name(g, "Ysum1")
        @test Ysum2 == get_tensor_by_name(g, "Ysum2")
        @test Ysum3 == get_tensor_by_name(g, "Ysum3")
        @test Ysum4 == get_tensor_by_name(g, "namefor_Ysum4")


    end
end

@testset "While" begin
    let
        sess = Session(Graph())
        i = constant(1)
        loop_sum = constant(0)
        res = @tf while i ≤ 10
            sq = i.^2
            [i=>i+1, loop_sum=>loop_sum+sq]
        end
        @test run(sess, res) == [11, 385]
    end
end
