using TensorFlow
using Base.Test

k = placeholder(Float32; shape=[10, 20, -1])
m = placeholder(Float32; shape=[10, 20, 30])
n = placeholder(Float32)

@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 3)
@test_throws BoundsError get_shape(k, 4)
@test_throws ErrorException get_shape(n, 1)


@test get.(get_shape(pack([m,m,m])).dims) == [3, 10, 20, 30]
@test get.(get_shape(pack([m,m,m],axis=2)).dims) == [10, 3, 20, 30]
@test get.(get_shape(pack([m,k])).dims) == [2, 10, 20, 30]
@test get.(get_shape(pack([k,m])).dims) == [2, 10, 20, 30]
@test get.(get_shape(pack([m,n])).dims) == [2, 10, 20, 30]
@test get_shape(pack([n,n])).rank_unknown
@test get.(get_shape(pack([k,k])).dims[1:3]) == [2, 10, 20]
@test isnull(get_shape(pack([k,k])).dims[4])

@test get.(get_shape(m+m).dims) == [10, 20, 30]
@test get_shape(m+n).rank_unknown
@test get_shape(m+k, 1) == 10
@test get_shape(m+k, 2) == 20
@test isnull(get_shape(m+k).dims[3])
