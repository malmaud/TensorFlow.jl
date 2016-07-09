sess=Session()


graph=read(open(joinpath(dirname(@__FILE__),"../test/graph.pb")))
status=extend_graph(sess, graph)
x=Tensor(Float32[1])
size(x,1)
x=Tensor(Float32[1])
size(x)
Array(x)
output=run(sess, ["x","y"], [Float32(10), Float32[2]], ["z"])

Array(output[1])[1]
