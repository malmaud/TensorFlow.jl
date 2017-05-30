module image

import ..TensorFlow
const tf = TensorFlow

import ..TensorFlow.Ops:
    decode_gif,
    decode_jpeg,
    encode_jpeg,
    decode_png,
    encode_png,
    resize_area,
    resize_bicubic,
    resize_bilinear,
    resize_nearest_neighbor,
    extract_glimpse,
    crop_and_resize,
    adjust_hue,
    adjust_saturation,
    draw_bounding_boxes,
    non_max_suppression,
    sample_distorted_bounding_box


@tf.op function flip_up_down(image; name=nothing)
    tf.Ops.reverse_v2(image, [1])
end

@tf.op function flip_left_right(image; name=nothing)
    tf.Ops.reverse_v2(image, [2])
end

end
