using Base.Test

let
    g = Graph()
    local i, j, k
    as_default(g) do
        @tf begin
            i = constant(1)
            j = constant(2)
            k = get_variable([], Float32)
        end
    end
    @test i == get_tensor_by_name(g, "i")
    @test j == get_tensor_by_name(g, "j")
    @test Tensor(k.var_node) == get_tensor_by_name(g, "k")
    @test Tensor(k.assign_node) == get_tensor_by_name(g, "k/Assign")
end

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
