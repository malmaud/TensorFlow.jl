module summary

export
SummaryWriter,
scalar,
audio,
histogram,
merge,
merge_all,
image

include("ops/summaries.jl")
include("summary_writer.jl")

using .summary_ops

end
