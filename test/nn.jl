using TensorFlow
using Base.Test

# conv2d_transpose
sess = Session(Graph())
value = placeholder(Float32, shape=[32, 10, 10, 3])
filter = placeholder(Float32, shape=[3, 3, 5, 3])
shape_ = placeholder(Int32, shape=[4])
y = nn.conv2d_transpose(value, filter, shape_, [1, 1, 1, 1])
run(sess, y, Dict(value=>randn(Float32, 32, 10, 10, 3),
                  filter=>randn(Float32, 3, 3, 5, 3),
                  shape_=>[32,10,10,5]))
