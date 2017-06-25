module summary_ops

export
scalar,
audio,
histogram,
merge_all,
image

import TensorFlow
const tf = TensorFlow

for (jl_func, op) in [
    (:scalar, :scalar_summary),
    (:audio, :audio_summary_v2),
    (:histogram, :histogram_summary),
    (:image, :image_summary)
    ]
    @eval @tf.op function $jl_func(args...; collections=[:Summaries], kwargs...)
        res = tf.Ops.$op(args...; kwargs...)
        foreach(c->tf.add_to_collection(c, res), collections)
        res
    end
end


const merge = tf.Ops.merge_summary

"""
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

end
