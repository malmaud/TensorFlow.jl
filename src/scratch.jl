include("ArrayFlow.jl")
g=ArrayFlow.Graph()
desc=ArrayFlow.NodeDescription(g, "Placeholder", "x")
desc["dtype"]=Float32
x=ArrayFlow.Node(desc)
desc=ArrayFlow.NodeDescription(g,"Placeholder","y")
desc["dtype"]=Float32
y=ArrayFlow.Node(desc)
desc=ArrayFlow.NodeDescription(g,"Add","z")
ArrayFlow.add_input(desc, ArrayFlow.Port(x,0))
ArrayFlow.add_input(desc,ArrayFlow.Port(y,0))
z=ArrayFlow.Node(desc)
sess=ArrayFlow.Session(g)

res=run(sess, [ArrayFlow.Port(x,0), ArrayFlow.Port(y,0)], [Float32(1), Float32(2)], [ArrayFlow.Port(z,0)], [])
