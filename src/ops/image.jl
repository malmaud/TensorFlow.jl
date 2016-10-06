module image

export
encode_jpeg,
decode_jpeg,
encode_png,
decode_png,
resize_images,
flip_up_down,
flip_left_right,
central_crop,
rgb_to_grayscale,
grayscale_to_rgb,
rgb_to_hsv,
hsv_to_rgb

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
function decode_jpeg(contents; channels=0, ratio=1, fancy_upscaling=true, try_recover_truncated=false, acceptable_fraction=1.0, name="DecodeJpeg")
    local desc
    with_op_name(name) do
        desc = NodeDescription("DecodeJpeg")
        add_input(desc, contents)
        desc["acceptable_fraction"] = Float32(acceptable_fraction)
        desc["channels"] = Int64(channels)
        desc["fancy_upscaling"] = fancy_upscaling
        desc["ratio"] = Int64(ratio)
        desc["try_recover_truncated"] = try_recover_truncated
    end
    Tensor(Operation(desc), 1)
end

function encode_jpeg(contents; format="", quality=95, progressive=true, optimize_size=false, chroma_downsampling=true, density_unit="in", x_density=300, y_density=300, xmp_metadata="", name="EncodeJpeg")
    local desc
    with_op_name(name) do
        desc = NodeDescription("EncodeJpeg")
        add_input(desc, contents)
        desc["quality"] = Int64(quality)
        desc["progressive"] = progressive
        desc["optimize_size"] = optimize_size
        desc["chroma_downsampling"] = chroma_downsampling
        desc["density_unit"] = density_unit
        desc["x_density"] = Int64(x_density)
        desc["y_density"] = Int64(y_density)
        desc["xmp_metadata"] = xmp_metadata
    end
    Tensor(Operation(desc), 1)
end

function decode_png(contents; channels=0, dtype=UInt8, name="DecodePng")
    local desc
    with_op_name(name) do
        desc = NodeDescription("DecodePng")
        add_input(desc, contents)
        desc["channels"] = Int64(channels)
        desc["dtype"] = dtype
    end
    Tensor(Operation(desc), 1)
end

function encode_png(image; compression::Integer=-1, name="EncodePng")
    local desc
    with_op_name(name) do
        desc = NodeDescription("EncodePng")
        add_input(desc, image)
        desc["compression"] = Int64(compression)
    end
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

function central_crop(image, crop_fraction; name="CentralCrop")
    local desc
    with_op_name(name) do
        desc = NodeDescription("CentralCrop")
        add_input(desc, image)
        add_input(desc, Float32(crop_fraction))
    end
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

for (jl_func_name, tf_func_name) in [
    (:rgb_to_grayscale, "rgb_to_grayscale"),
    (:grayscale_to_rgb, "grayscale_to_rgb"),
    (:hsv_to_rgb, "hsv_to_rgb"),
    (:rgb_to_hsv, "rgb_to_hsv")]
    @eval function $jl_func_name(image; name=$tf_func_name)
        local desc
        with_op_name(name) do
            desc = NodeDescription($tf_func_name)
            add_input(desc, image)
        end
        Tensor(Operation(desc), 1)
    end
end

end
