using TensorFlow
using Base.Test

graph = Graph()
sess = Session(graph)
mktempdir() do tmpdir
    @test_nowarn writer = train.SummaryWriter(tmpdir)
    x = constant([1,2])
    x_summary = scalar_summary(["x", "y"], x)
    hist_summary = histogram_summary("z", randn(10))
    summaries = merge_all_summaries()
    summary_pb = run(sess, summaries)
    @test_nowarn write(writer, get_def_graph())
    @test_nowarn write(writer, summary_pb)
    @test_nowarn close(writer)
end
