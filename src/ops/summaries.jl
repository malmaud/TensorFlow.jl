module summary_ops

export
scalar,
audio,
histogram,
merge_all,
image

import TensorFlow
const tf = TensorFlow

const scalar = tf.Ops.scalar_summary
const audio = tf.Ops.audio_summary_v2
const histogram = tf.Ops.histogram_summary
const image = tf.Ops.image_summary
const merge = tf.Ops.merge_summary

"""
Merges all summaries collected in the default graph.

Args:
  `key`: `GraphKey` used to collect the summaries.  Defaults to
          `:summaries`

Returns:
  If no summaries were collected, returns nothing.  Otherwise returns a scalar
  `Tensor` of type `String` containing the serialized `Summary` protocol
  buffer resulting from the merging.
"""
function merge_all(key=:Summaries)
    merge(tf.get_collection(key), collections=[])
end

end
