sess=Session()

# status=extend_graph(sess, graph)
# output=run(sess, "z", Dict("x"=>Float32(10.0), "y"=>Float32[3,4]))
# output

x=placeholder(TF_FLOAT)
y=placeholder(TF_FLOAT)
z=x+y
extend_graph(sess)

output=run(sess, z, Dict(x=>Float32(10.0), y=>Float32[3,4]))
output

x.o[:name]
