
using TensorFlow

sess = Session(Graph())
i = constant(1, name="a")
try
    end_i = TensorFlow.while_loop((i->i <= 10), i->i+1, [i])

    run(sess, end_i)
catch err
    @show err
end
