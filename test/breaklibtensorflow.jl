using TensorFlow
import TensorFlow: get_tensors, name_scope, with_frame

sess = Session(Graph())
i = constant(1; name="ff")


variables = [i]


g = Graph()
def_graph = get_def_graph()
g.op_context = def_graph.op_context
g.name_idx = def_graph.name_idx
g.collections = def_graph.collections

as_default(g) do
    with_frame(2, true, false) do
        context = get_def_graph().op_context.while_context[end]
        enter_op = Ops.enter(i, frame_name=context.context_name)
    end
end

####################



sess=Session(Graph())
j=constant(31)
g=Graph()
c=as_default(g) do
    c = 2*j
end
