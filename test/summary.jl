using TensorFlow
const tf = TensorFlow
using Base.Test

graph = Graph()
sess = Session(graph)
mktempdir() do tmpdir
    writer = tf.summary.FileWriter(tmpdir)
    x = constant([1,2])
    x_summary = tf.summary.scalar(["x", "y"], x)
    hist_summary = tf.summary.histogram("z", randn(10))
    summaries = tf.summary.merge_all()
    summary_pb = run(sess, summaries)
    write(writer, get_def_graph())
    write(writer, summary_pb)
    close(writer)
end

get_collection(graph, :Summaries)
