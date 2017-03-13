import Base: @deprecate, depwarn

@deprecate scalar_summary(args...; kwargs...) summary.scalar(args...; kwargs...)
@deprecate audio_summary(args...; kwargs...) summary.audio(args...; kwargs...)
@deprecate histogram_summary(args...; kwargs...) summary.histogram(args...; kwargs...)
@deprecate image_summary(args...; kwargs...) summary.image(args...; kwargs...)
@deprecate merge_summary(args...; kwargs...) summary.merge(args...; kwargs...)
@deprecate merge_all_summaries(args...; kwargs...) summary.merge_all(args...; kwargs...)

@deprecate stack(args...; kwargs...) pack(args...; kwargs...)
@deprecate unstack(args...; kwargs...) unpack(args...; kwargs...)
