module image

export
decode_jpeg,
decode_png,
resize_images,
flip_up_down,
flip_left_right

import ..TensorFlow: NodeDescription, get_def_graph, get_name, add_input, Operation, pack, convert_number, AbstractOperation, Tensor, with_op_name, constant

"""
`function decode_jpeg(contents, channels=1, ratio=1, fancy_upscaling=true, try_recover_truncated=false, acceptable_fraction=1.0)`

Decode a JPEG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the JPEG-encoded image.
*   1: output a grayscale image.
*   3: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
downscaling the image later.

Args:
  contents: A `Tensor` of type `String`. 0-D.  The JPEG-encoded image.
  channels: An optional `int`. Defaults to `0`.
    Number of color channels for the decoded image.
  ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
  fancy_upscaling: An optional `Bool`. Defaults to `true`.
    If true use a slower but nicer upscaling of the
    chroma planes (yuv420/422 only).
  try_recover_truncated: An optional `Bool`. Defaults to `false`.
    If true try to recover an image from truncated input.
  acceptable_fraction: An optional `Float32`. Defaults to `1`.
    The minimum required fraction of lines before a truncated
    input is accepted.
  name: A name for the operation (optional).

Returns:
  A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`.
"""
function decode_jpeg(contents; channels=0, ratio=1, fancy_upscaling=true, try_recover_truncated=false, acceptable_fraction=1.0, name="")
    desc = NodeDescription(get_def_graph(), "DecodeJpeg", get_name(name))
    add_input(desc, contents)
    desc["acceptable_fraction"] = Float32(acceptable_fraction)
    desc["channels"] = Int64(channels)
    desc["fancy_upscaling"] = fancy_upscaling
    desc["ratio"] = Int64(ratio)
    desc["try_recover_truncated"] = try_recover_truncated
    Tensor(Operation(desc), 1)
end

function decode_png(contents; channels=0, dtype=UInt8, name="")
    desc = NodeDescription(get_def_graph(), "DecodePng", get_name(name))
    add_input(desc, contents)
    desc["channels"] = Int64(channels)
    desc["dtype"] = dtype
    Tensor(Operation(desc), 1)
end


@enum ResizeMethod BILINEAR NEAREST_NEIGHBOR BICUBIC AREA

function resize_images(images, new_height, new_width; method=BILINEAR, align_corners=false, name="")
    op_names = Dict(BILINEAR=>"ResizeBilinear", BICUBIC=>"ResizeBicubic")
    desc = NodeDescription(get_def_graph(), op_names[method], get_name(name))
    add_input(desc, images)
    dims = pack([convert_number(Int32,new_height), convert_number(Int32,new_width)])
    add_input(desc, dims)
    desc["align_corners"] = align_corners
    Tensor(Operation(desc), 1)
end

function flip_up_down(image; name="FlipUpDown")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Reverse")
        add_input(desc, image)
        dims = Tensor([true, false, false])
        add_input(desc, dims)
    end
    Tensor(Operation(desc), 1)
end

function flip_left_right(image; name="FlipLeftRight")
    local desc
    with_op_name(name) do
        desc = NodeDescription("Reverse")
        add_input(desc, image)
        dims = Tensor([false, true, false])
        add_input(desc, dims)
    end
    Tensor(Operation(desc), 1)
end

end
