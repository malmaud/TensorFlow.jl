import Base: @deprecate, depwarn

@deprecate scalar_summary(args...; kwargs...) summary.scalar(args...; kwargs...)
@deprecate audio_summary(args...; kwargs...) summary.audio(args...; kwargs...)
@deprecate histogram_summary(args...; kwargs...) summary.histogram(args...; kwargs...)
@deprecate image_summary(args...; kwargs...) summary.image(args...; kwargs...)
@deprecate merge_summary(args...; kwargs...) summary.merge(args...; kwargs...)
@deprecate merge_all_summaries(args...; kwargs...) summary.merge_all(args...; kwargs...)

@deprecate pack(args...; kwargs...) stack(args...; kwargs...)
@deprecate unpack(args...; kwargs...) unstack(args...; kwargs...)

@deprecate initialize_all_variables() global_variables_initializer()
