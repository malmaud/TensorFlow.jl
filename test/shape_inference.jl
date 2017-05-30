using TensorFlow
using Base.Test

k = placeholder(Float32; shape=[10, 20, -1])
m = placeholder(Float32; shape=[10, 20, 30])
n = placeholder(Float32)
i = placeholder(Int32; shape=[])

@testset "placeholder" begin
    @test get_shape(k) == TensorShape([10, 20, -1])
    @test get_shape(m) == TensorShape([10, 20, 30])
    @test get_shape(n) == TensorShape(nothing)
    @test get_shape(i) == TensorShape([])

    @test get_shape(k,2) == 20
    @test_throws ErrorException get_shape(k, 3)
    @test_throws BoundsError get_shape(k, 4)
    @test_throws ErrorException get_shape(n, 1)
end

@testset "Transpose/Permutedims" begin
    #Constant propergation in shape_inference is not yet up to the task for this
    #@test_broken get_shape(k') == get_shape(transpose(k)) == TensorShape([-1, 20, 10])
    #@test_broken get_shape(m') == get_shape(transpose(m)) == TensorShape([30, 20, 10])

    @test get_shape(n') == get_shape(transpose(n)) == TensorShape(nothing)

    @test get_shape(permutedims(m, [3,1,2])) == TensorShape([30, 10, 20])
    @test get_shape(permutedims(n, [3,1,2,4])) == TensorShape([-1, -1, -1, -1])

end


@testset "Arithmetic" begin
    @test get_shape(-k) == get_shape(k)
    @test get_shape(k+1) == get_shape(k)
    @test get_shape(k-1) == get_shape(k)

    @test get_shape(1+k) == get_shape(k)
    @test get_shape(1-k) == get_shape(k)
    @test get_shape(2*k) == get_shape(k)


    @test get_shape(m+m) == TensorShape([10, 20, 30])
    @test get_shape(m+n).rank_unknown
    @test get_shape(m+k) == TensorShape([10, 20, -1])
end

@testset "Find (i.e Where)" begin
    @test get_shape(find(placeholder(Bool; shape=[10, 20, 30]))) == TensorShape([-1,3])
    @test get_shape(find(placeholder(Bool; shape=[10, 20, -1]))) == TensorShape([-1,3])
    @test get_shape(find(placeholder(Bool))) == TensorShape(nothing)
end

@testset "Stack/Unstack" begin
    #stack
    @test get_shape(stack([m,m,m])) == TensorShape([3, 10, 20, 30])
    @test get_shape(stack([m,m,m],axis=2)) == TensorShape([10, 3, 20, 30])
    @test get_shape(stack([m,k])) == TensorShape([2, 10, 20, 30])
    @test get_shape(stack([k,m])) == TensorShape([2, 10, 20, 30])
    @test get_shape(stack([m,n])) == TensorShape([2, 10, 20, 30])
    @test get_shape(stack([n,n])).rank_unknown
    @test get_shape(stack([k,k])) == TensorShape([2, 10, 20, -1])

    ## unstack
    for ii in 1:10
        @test get_shape(unstack(m)[ii]) == TensorShape([20, 30])
    end

    for ii in 1:20
        @test get_shape(unstack(k, axis=2)[ii]) == TensorShape([10, -1])
    end

    for ii in 1:3
        @test get_shape(unstack(n, num=3)[ii]) == TensorShape(nothing)
    end


    ### stack/unstack
    @test get_shape(stack(unstack(m))) == get_shape(m)
    @test get_shape(stack(unstack(k))) == get_shape(k)
end



@testset "Concat" begin
    @test get_shape(cat(2, m,m)) == TensorShape([10, 40, 30])
    @test get_shape(cat(2, m,k))  == TensorShape([10, 40, 30])
    @test get_shape(cat(2, k,m))  == TensorShape([10, 40, 30])
    @test get_shape(cat(3, m,k))  == TensorShape([10,20, -1])
end


@testset "GatherNd" begin
    @test get_shape(gather_nd(m, [3])) == TensorShape([20, 30]) #1

    @test get_shape(gather_nd(m, [5,6,6])) == TensorShape([]) #3
    @test get_shape(gather_nd(m, [5 6 6])) == TensorShape([1])#1x3
    @test get_shape(gather_nd(m, [5 6 6]')) == TensorShape([3, 20, 30])#3x1

    @test get_shape(gather_nd(m, [2 5; 2 6; 2 7])) == TensorShape([3, 30]) #2x3
    @test get_shape(gather_nd(m, [2 2 2; 5 6 7])) == TensorShape([2]) #3x2

    @test get_shape(gather_nd(m, [5,6])) == TensorShape([30]) #2
    @test get_shape(gather_nd(m, [5 6]')) == TensorShape([2, 20, 30]) #2x1

    @test get_shape(gather_nd(m, reshape([3], (1,1)))) == TensorShape([1, 20, 30]) #1x1
    @test get_shape(gather_nd(m, reshape([3], (1,1,1)))) == TensorShape([1, 1, 20, 30]) #1x1x1
end

@testset "ScatterNd" begin
    @test get_shape(scatter_nd([2], [6], [4])) == TensorShape([4])
    @test get_shape(scatter_nd([5 4 2 8]', [9, 10, 11, 12], [8])) == TensorShape([8])
    @test get_shape(scatter_nd([5 3]', [9 9; 10 10], [6,2])) == TensorShape([6, 2])

    @test get_shape(scatter_nd([5 3]', [9 9; 10 10], TensorShape([6,2]))) == TensorShape([6, 2])
end


@testset "ExpandDims" begin
    @test get_shape(expand_dims(m, 1)) == TensorShape([1, 10, 20, 30])
    @test get_shape(expand_dims(m, 2)) == TensorShape([10, 1, 20, 30])
    @test get_shape(expand_dims(m, 3)) == TensorShape([10, 20, 1, 30])
    @test get_shape(expand_dims(m, 4)) == TensorShape([10, 20, 30, 1])
    @test get_shape(expand_dims(m, 0)) == TensorShape([10, 20, 30, 1])
    @test get_shape(expand_dims(m, -1)) == TensorShape([10, 20, 1, 30])
    @test get_shape(expand_dims(m, i)) == TensorShape([-1, -1, -1, -1])
    @test get_shape(expand_dims(n, 2)) == TensorShape(nothing)
end




@testset "Squeeze" begin
    let
        x = placeholder(Float64, shape=[5, 1, 4, 1, 3])
        y = squeeze(x, [2, 4])
        @test get_shape(y) == TensorShape([5, 4, 3])
    end
end

@testset "Slice" begin
    let
        x = placeholder(Float64, shape=[2, 3])
        y = TensorFlow.slice(x, [1, 2], [-1, 2])
        @test get_shape(y) == TensorShape([2, 2])
    end
end

@testset "nn.softmax_cross_entropy_with_logits" begin
    a = placeholder(Float32; shape=[10, 20])
    b = placeholder(Float32)
    c = placeholder(Float32; shape=[-1, 20])
    d = placeholder(Float32; shape=[100, 20])

    for labels in [a,b,c]
        for logits in [a,b,c]
            res = get_shape(nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            if labels !==a && logits !== a
                @test res == TensorShape([-1])
            else
                @test res == TensorShape([10])
            end
        end
    end

    @test_throws DimensionMismatch get_shape(nn.softmax_cross_entropy_with_logits(logits=a, labels=d))
end

@testset "round" begin
    for var in [k,m,n,i]
        @test get_shape(round(var)) == get_shape(var)
        @test get_shape(round(Int, var)) == get_shape(var)
    end
end

@testset "squared_difference" begin
    @test get_shape(squared_difference([1,2], 3)) == TensorShape([2])
end

@testset "unsorted_segment_sum" begin
    @test isnull(get_shape(unsorted_segment_sum(m, placeholder(Int64), placeholder(Int32))).dims[1])
    @test get_shape(unsorted_segment_sum(m, placeholder(Int64), Int32(5))).dims[1].value==5
end


