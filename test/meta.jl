using Test
using TensorFlow

@testset "Registering Ops" begin

    @testset "Registered types" begin
        # Only the 2 arg version should be registered
        @test TensorFlow.is_registered_op(TensorFlow.FIFOQueue, 32, [Int,Float32]) == TensorFlow.RegisteredOp()
        @test TensorFlow.is_registered_op(TensorFlow.FIFOQueue, 32) == TensorFlow.NotRegisteredOp()
        @test TensorFlow.is_registered_op(TensorFlow.FIFOQueue) == TensorFlow.NotRegisteredOp()
    end

    @testset "Unregistered types" begin
        @test TensorFlow.is_registered_op(nn.rnn_cell.GRUCell) == TensorFlow.NotRegisteredOp()
        @test TensorFlow.is_registered_op(nn.rnn_cell.GRUCell, 32) == TensorFlow.NotRegisteredOp()
    end

    @testset "Registered Functions" begin
        @test TensorFlow.is_registered_op(typeof(nn.softmax), [1,0]) == TensorFlow.RegisteredOp()
        @test TensorFlow.is_registered_op(typeof(nn.softmax)) == TensorFlow.NotRegisteredOp()
        @test TensorFlow.is_registered_op(typeof(nn.softmax), [1,0], 2) == TensorFlow.NotRegisteredOp()


        @test TensorFlow.is_registered_op(typeof(placeholder), Int) == TensorFlow.RegisteredOp()
        @test TensorFlow.is_registered_op(typeof(placeholder)) == TensorFlow.NotRegisteredOp()

        @test TensorFlow.is_registered_op(typeof(placeholder), Int) == TensorFlow.RegisteredOp()
        @test TensorFlow.is_registered_op(typeof(placeholder)) == TensorFlow.NotRegisteredOp()
    end

    @testset "Unregistered Functions" begin
        @test TensorFlow.is_registered_op(typeof(string)) == TensorFlow.NotRegisteredOp()
        @test TensorFlow.is_registered_op(typeof(string),3 ) == TensorFlow.NotRegisteredOp()
    end
end

@testset "Types with typeparams" begin
    @tf begin
        fooo = nn.rnn_cell.DropoutWrapper(nn.rnn_cell.GRUCell(3), Tensor(0.2))
        @test true # Above line would have errored if @tf macro was broken
    end
end

@testset "colon #448" begin
    @tf for ii in 1:10
    end
    @test true # would have errored if this was broken
end

@testset "While" begin
    @test_broken let
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

@testset "Negating and @tf" begin
    a = constant(1)    
    @tf b = -a
    @test true # Above line would have errored if unary - didn't work with @tf
end

@testset "Naming Big Demo" begin
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


