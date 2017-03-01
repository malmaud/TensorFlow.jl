using TensorFlow
using Base.Test

import TensorFlow.ShapeInference: TensorShape
k = placeholder(Float32; shape=[10, 20, -1])
m = placeholder(Float32; shape=[10, 20, 30])
n = placeholder(Float32)
i = placeholder(Int32; shape=[])

@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 3)
@test_throws BoundsError get_shape(k, 4)
@test_throws ErrorException get_shape(n, 1)

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
@test get_shape(k*2) == get_shape(k)

@test get_shape(1+k) == get_shape(k)
@test get_shape(1-k) == get_shape(k)
@test get_shape(2*k) == get_shape(k)
