sess=Session()


graph=read(open(joinpath(dirname(@__FILE__),"../test/graph.pb")))
status=extend_graph(sess, graph)
output=run(sess, "z", Dict("x"=>Float32[1,2], "y"=>Float32[3,4]))
output
