module summary_ops

export
scalar,
audio,
histogram,
merge_all,
image

import TensorFlow
const tf = TensorFlow
import TensorFlow: with_op_name, NodeDescription, add_input, Tensor,
                   add_to_collection, Operation, get_collection, @op

"""
Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated
summary has a summary value for each tag-value pair in `tags` and `values`.

Args:
* `tags`: A `string` `Tensor`.  Tags for the summaries.
* `values`: A real numeric `Tensor`.  Values for the summaries.
* `collections`: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.
"""
@op function scalar(tags, values; collections=[:Summaries], name=nothing)
    local desc
    with_op_name(name, "ScalarSummary") do
        desc = NodeDescription("ScalarSummary")
        add_input(desc, Tensor(tags))
        add_input(desc, Tensor(values))
    end
    t = Tensor(Operation(desc))
    for collection in collections
        add_to_collection(collection, t)
    end
    return t
end

"""
Outputs a `Summary` protocol buffer with audio.

Args:
* `tag`: A `string` `Tensor`.  Tag for the summary.
* `tensor`: A real numeric `Tensor`.  Values for the summaries.
* `sample_rate`: The sample rate of the signal in Hertz.
* `max_outputs`: Maximum number of batch outputs to generate audio for.
* `collections`: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.
"""
@op function audio(tag, tensor, sample_rate; max_outputs=3, collections=[:Summaries], name=nothing)
    local desc
    with_op_name(name, "AudioSummary") do
        desc = NodeDescription("AudioSummary")
        add_input(desc, Tensor(tag))
        add_input(desc, Tensor(tensor))
        desc["sample_rate"] = sample_rate
        desc["max_outputs"] = max_outputs
    end
    t = Tensor(Operation(desc))
    for collection in collections
        add_to_collection(collection, t)
    end
    return t
end

"""
Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

Args:
*  `tag`: A `string` `Tensor`. 0-D.  Tag to use for the summary value.
*  `values`: A real numeric `Tensor`. Any shape. Values to use to
    build the histogram.
*  `collections`: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  `name`: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.
"""
@op function histogram(tag, values; collections=[:Summaries], name=nothing)
    local desc
    with_op_name(name, "HistogramSummary") do
        desc = NodeDescription("HistogramSummary")
        add_input(desc, Tensor(tag))
        add_input(desc, Tensor(values))
    end
    t = Tensor(Operation(desc))
    foreach(c->add_to_collection(c, t), collections)
    t
end

"""
Creates a `Summary` protocol buffer containing the unions of all the summaries
in `inputs`.

Args:
* `inputs`: A list of `string` `Tensor`s containing serialized `Summary` buffers.
*  `collections`: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  `name`: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer resulting from the merge.
"""
@op function merge(inputs; collections=[:Summaries], name=nothing)
    local desc
    with_op_name(name, "MergeSummary") do
        desc = NodeDescription("MergeSummary")
        add_input(desc, inputs)
    end
    t = Tensor(Operation(desc))
    for collection in collections
        add_to_collection(collection, t)
    end
    return t
end

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
    merge(get_collection(key), collections=[])
end

"""
Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `UInt8` values are unchanged.  The op uses two different
normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `String`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

Args:
  tag: A scalar `Tensor` of type `string`. Used to build the `tag`
    of the summary values.
  tensor: A 4-D `UInt8` or `Float32` `Tensor` of shape `[batch_size, height,
    width, channels]` where `channels` is 1, 3, or 4.
  max_images: Max number of batch elements to generate images for.
  collections: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
  name: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.
"""
@op function image(tag, tensor; max_images=3, collections=[:Summaries], name=nothing)
    local desc
    with_op_name(name, "ImageSummary") do
        desc = NodeDescription("ImageSummary")
        add_input(desc, tag)
        add_input(desc, tensor)
        desc["max_images"] = Int64(max_images)
    end
    t = Tensor(Operation(desc))
    foreach(c->add_to_collection(c, t), collections)
    t
end

end
