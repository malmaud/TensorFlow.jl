using TensorFlow
using Base.Test

sess = TensorFlow.Session(TensorFlow.Graph())

one_tens = ones(Tensor, (5,5))

@test [1, 2] == run(sess, cast(constant([1.8, 2.2]), Int))

@test ones(25) == run(sess, reshape(one_tens, 25))


@test Int32[5,5,1] == run(sess, TensorFlow.shape(stack(split(2, 5, one_tens), axis=1)))

@test ones(Float32, 5,5) == run(sess, stack(unstack(one_tens, num=5)))
# @test ones(Float32, 5,5) == run(sess, pack(unpack(one_tens, axis=1)))

@test ones(1,5,5) == run(sess, expand_dims(one_tens, 1))
@test ones(5,1,5) == run(sess, expand_dims(one_tens, 2))
@test ones(5,1,5) == run(sess, expand_dims(one_tens, -1))
@test ones(5,5,1) == run(sess, expand_dims(one_tens, 0))

@test 2 == run(sess, rank(one_tens))

@test ones(10,5) == run(sess, tile(one_tens, [2; 1]))

@test ones(Float32, 4,3) == run(sess, transpose(ones(Tensor, (3, 4))))
@test ones(Float32, 4,3,2) == run(sess, permutedims(ones(Tensor, (4, 2, 3)), [1, 3, 2]))

@test hcat(ones(Float32, 5,5), zeros(Float32, 5)) == run(sess, pad(one_tens, [0 0; 0 1]))

@test Float32[1.; 0.; 0.; 0.; 0.] == run(sess, one_hot(1, 5))

a = Tensor(collect(1:5))
result = run(sess, shuffle(a))
for i in 1:5
    @test i ∈ result
end


# Test `squeeze()` works when given explicit dimensions, fails on incorrect explicit dimensions,
# and works when given no explicit dimension
sq_ones = ones(Tensor, (10, 1, 5, 1))
@test size(run(sess, squeeze(sq_ones))) == (10,5)
@test size(run(sess, squeeze(sq_ones,[2,4]))) == (10,5)
@test size(run(sess, squeeze(sq_ones,[2]))) == (10,5,1)
@test_throws TensorFlow.TFException run(sess, squeeze(sq_ones,[1]))

#######################################################################
# getindex related methods (getindex overload and the methods behind it)

# Test values
x_jl = [10x+y for x in 1:5, y in 1:7]
x = constant(x_jl)
w_jl = [100x+10y+z for x in 1:5, y in 1:7, z in 1:3]
w = constant(w_jl)
y = constant([1, 2, 3])

### Mask (bool array)

mask = constant([true, false, true])
@test run(sess, boolean_mask(y,mask))  == run(sess, y[mask]) == [1, 3]

### Gather (int/ int array) / Index

@test ones(Float32, 2, 5) == run(sess, gather(one_tens, [1, 2]))
@test run(sess, y[[1, 3]]) == [1, 3]
@test run(sess, y[2]) == 2

### Gather-nd / Cartean Index/Slice
@test run(sess, gather_nd(x, [2, 3])) == x_jl[2,3]
@test run(sess, x[2,3]) == x_jl[2,3]

@test run(sess, gather_nd(x, [3])) == x_jl[3,:]

@test run(sess, gather_nd(x, [1 1; 2 3])) == [x_jl[1,1], x_jl[2,3]]
@test run(sess, gather_nd(x, [1 2]')) == [x_jl[1,:]'; x_jl[2,:]']


### Slice
# to do make sure we slice the right indices
@test ones(Float32, 5).' == run(sess, TensorFlow.slice(one_tens, [0, 0], [1, -1]))

### ScatterNd

@test run(sess, TensorFlow.scatter_nd([2], [6], [4])) == [0, 6, 0, 0]
@test run(sess, TensorFlow.scatter_nd([5 4 2 8]', [9, 10, 11, 12], [8])) == [0, 11, 0, 10, 9, 0, 0, 12]
@test run(sess, TensorFlow.scatter_nd([5 3]', [9 9; 10 10], [6,2])) == [0 0; 0 0; 10 10; 0 0; 9 9; 0 0]


############
# Check it gather_nd can make a network
sess2 = Session(Graph())
embs = get_variable("tt2", (10,10), Float64)
vals = gather_nd(embs,[2])
cost = reduce_sum(vals)
optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
run(sess2, initialize_all_variables())
@test length(run(sess2, vals)) == 10

# Check concat can make a network Issue #147
sess3 = Session(Graph())
x1 = constant(rand(20,10))
x2 = get_variable("x2", (50,10), Float64)
xs = concat([x1,x2], 1)
cost = reduce_sum(xs)
optimizer = train.minimize(train.AdamOptimizer(0.1), cost)
run(sess3, initialize_all_variables())
@test size(run(sess3, xs)) == (70, 10)
