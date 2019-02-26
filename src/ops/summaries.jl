module summary_ops

export
scalar,
audio,
histogram,
merge_all,
image,
@scalar

import TensorFlow
using MacroTools
const tf = TensorFlow

for (jl_func, op) in [
    (:scalar, :scalar_summary),
    (:audio, :audio_summary_v2),
    (:histogram, :histogram_summary),
    (:image, :image_summary)
    ]
    @eval @tf.op function $jl_func(args...; collections=[:Summaries], step=0, kwargs...)
        res = tf.Ops.$op(args...; kwargs...)
        if tf.in_eager_mode()
          tf.summary.record_summary(tf.item(res), step=step)
        else
          foreach(c->tf.add_to_collection(c, res), collections)
          return res
        end
    end

    # Set the documentation of the summary function to the same as the
    # documentation of the underlying TensorFlow op
    @eval @doc(@doc(tf.Ops.$op), $jl_func)
end


const merge = tf.Ops.merge_summary

"""
    merge_all(key=:Summaries)

Merges all summaries collected in the default graph.

Args:
  `key`: `GraphKey` used to collect the summaries.  Defaults to
          `:Summaries`

Returns:
  If no summaries were collected, returns nothing.  Otherwise returns a scalar
  `Tensor` of type `String` containing the serialized `Summary` protocol
  buffer resulting from the merging.
"""
function merge_all(key=:Summaries)
    tensors = tf.get_collection(key)
    merge(tensors)
end

macro scalar(f, args...)
  quote
    scalar($(string(f)), $(esc(f)); $(esc.(args)...))
  end
end

end
