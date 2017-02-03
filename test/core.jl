
k = placeholder(Float32; shape=[10,20, -1])
@test get_shape(k,2) == 20
@test_throws ErrorException get_shape(k, 3)
@test_throws BoundsError get_shape(k, 4)

@test_throws ErrorException get_shape(placeholder(Float32), 1)

