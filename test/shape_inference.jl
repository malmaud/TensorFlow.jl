using TensorFlow
using Base.Test

import TensorFlow.ShapeInference: TensorShape
k = placeholder(Float32; shape=[10, 20, -1])
m = placeholder(Float32; shape=[10, 20, 30])
n = placeholder(Float32)
i = placeholder(Int32; shape=[])



@test get_shape(k) == TensorShape([10, 20, -1])
@test get_shape(m) == TensorShape([10, 20, 30])
@test get_shape(n) == TensorShape(nothing)
@test get_shape(i) == TensorShape([])

@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 3)
@test_throws BoundsError get_shape(k, 4)
@test_throws ErrorException get_shape(n, 1)

## Find (i.e Where)
@test get_shape(find(placeholder(Bool; shape=[10, 20, 30]))) == TensorShape([-1,3])
@test get_shape(find(placeholder(Bool; shape=[10, 20, -1]))) == TensorShape([-1,3])
@test get_shape(find(placeholder(Bool))) == TensorShape(nothing)

## Pack
@test get_shape(pack([m,m,m])) == TensorShape([3, 10, 20, 30])
@test get_shape(pack([m,m,m],axis=2)) == TensorShape([10, 3, 20, 30])
@test get_shape(pack([m,k])) == TensorShape([2, 10, 20, 30])
@test get_shape(pack([k,m])) == TensorShape([2, 10, 20, 30])
@test get_shape(pack([m,n])) == TensorShape([2, 10, 20, 30])
@test get_shape(pack([n,n])).rank_unknown
@test get_shape(pack([k,k])) == TensorShape([2, 10, 20, -1])

## Unpack
for ii in 1:10
    @test get_shape(unpack(m)[ii]) == TensorShape([20, 30])
end

for ii in 1:20
    @test get_shape(unpack(k, axis=2)[ii]) == TensorShape([10, -1])
end

for ii in 1:3
    @test get_shape(unpack(n, num=3)[ii]) == TensorShape(nothing)
end



### Pack/UnPack
@test get_shape(pack(unpack(m))) == get_shape(m)
@test get_shape(pack(unpack(k))) == get_shape(k)


## Add
@test get_shape(m+m) == TensorShape([10, 20, 30])
@test get_shape(m+n).rank_unknown
@test get_shape(m+k) == TensorShape([10, 20, -1])

## Concat
@test get_shape(cat(2, m,m)) == TensorShape([10, 40, 30])
@test get_shape(cat(2, m,k))  == TensorShape([10, 40, 30])
@test get_shape(cat(2, k,m))  == TensorShape([10, 40, 30])
@test get_shape(cat(3, m,k))  == TensorShape([10,20, -1])

## GatherNd
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


## ScatterNd
@test get_shape(scatter_nd([2], [6], [4])) == TensorShape([4])
@test get_shape(scatter_nd([5 4 2 8]', [9, 10, 11, 12], [8])) == TensorShape([8])
@test get_shape(scatter_nd([5 3]', [9 9; 10 10], [6,2])) == TensorShape([6, 2])


@test get_shape(scatter_nd([5 3]', [9 9; 10 10], TensorShape([6,2]))) == TensorShape([6, 2])



## ExpandDims
@test get_shape(expand_dims(m, 1)) == TensorShape([1, 10, 20, 30])
@test get_shape(expand_dims(m, 2)) == TensorShape([10, 1, 20, 30])
@test get_shape(expand_dims(m, 3)) == TensorShape([10, 20, 1, 30])
@test get_shape(expand_dims(m, 4)) == TensorShape([10, 20, 30, 1])
@test get_shape(expand_dims(m, 0)) == TensorShape([10, 20, 30, 1])
@test get_shape(expand_dims(m, -1)) == TensorShape([10, 20, 1, 30])
@test get_shape(expand_dims(m, i)) == TensorShape([-1, -1, -1, -1])
@test get_shape(expand_dims(n, 2)) == TensorShape(nothing)


## Basic Operations
@test get_shape(-k) == get_shape(k)
@test get_shape(k+1) == get_shape(k)
@test get_shape(k-1) == get_shape(k)

@test get_shape(1+k) == get_shape(k)
@test get_shape(1-k) == get_shape(k)
@test get_shape(2*k) == get_shape(k)

## Squeeze
let
    x = placeholder(Float64, shape=[5, 1, 4, 1, 3])
    y = squeeze(x, [2, 4])
    @test get_shape(y) == TensorShape([5, 4, 3])
end

## Slice
let
    x = placeholder(Float64, shape=[2, 3])
    y = TensorFlow.slice(x, [0, 1], [-1, 2])
    @test get_shape(y) == TensorShape([2, 2])
end
