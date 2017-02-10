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

Decode a JPEG-encoded image to a `uint8` `Tensor`.

The attribute `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   `0`: Use the number of channels in the JPEG-encoded image.
*   `1`: output a grayscale image.
*   `3`: output an RGB image.

If needed, the JPEG-encoded image is transformed to match the requested number
of color channels.

The attr `ratio` allows downscaling the image by an integer factor during
decoding.  Allowed values are: `1`, `2`, `4`, and `8`.  This is much faster than
downscaling the image later.

Args:
  `contents`: A `Tensor` of type `String`. 0-D.  The JPEG-encoded image.
  `channels`: An optional `int`. Defaults to `0`.
    Number of color channels for the decoded image.
  `ratio`: An optional `int`. Defaults to `1`. Downscaling ratio.
  `fancy_upscaling`: An optional `Bool`. Defaults to `true`.
    If true use a slower but nicer upscaling of the
    chroma planes (yuv420/422 only).
  `try_recover_truncated`: An optional `Bool`. Defaults to `false`.
    If true try to recover an image from truncated input.
  `acceptable_fraction`: An optional `Float32`. Defaults to `1`.
    The minimum required fraction of lines before a truncated
    input is accepted.
  `name`: A name for the operation (optional).

Returns:
  A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`.
"""
function decode_jpeg(contents; channels=0, ratio=1, fancy_upscaling=true, try_recover_truncated=false, acceptable_fraction=1.0, name=nothing)
    local desc
    with_op_name(name, "DecodeJpeg") do
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

"""
    encode_jpeg(contents; format="", quality=95, progressive=true, optimize_size=false, chroma_downsampling=true, density_unit="in", x_density=300, y_density=300, xmp_metadata="", name="EncodeJpeg")

Encode a 3D `UInt8` `Tensor` `contents` to a JPEG image. `contents` has shape `[height, width, channels]`.

Args:
*  `contents`: A 3D `UInt8` `Tensor`.
*  `format`: Per-pixel image format. Options are `""` (default), `"grayscale"`, or `"rgb"`.
*  `quality`: Compression quality from `0` to `100`. Default is `95`. A larger number is better but slower.
*  `progressive`: If `true` (default is `false`), creates a JPEG that loads progressively.
*  `optimize_size`: If `true` (default is `false`), use extra CPU/RAM to make the image filesize smaller without affecting image quality.
*  `chroma_downsampling`: Defaults to `true`.
*  `density_unit`: Optional `string`, defaults to `"in"` for inches. `"cm"` can be used to specify centimeters.
*  `x_density`: Optional `Int`, defaults to `300`. Horizontal pixels per `density_unit`.
*  `y_density`: Optional `Int`, defaults to `300`. Vertical pixels per `density_unit`.
*  `xmp_metadata`: Optional `string`, defaults to `""`. If not empty, embed the string as XMP metadata in the image header.
*  `name`: A name for the operation (optional).

Returns:
  A `Tensor` of type `string`. Zero dimensional JPEG-encoded image.
"""
function encode_jpeg(contents; format="", quality=95, progressive=true, optimize_size=false, chroma_downsampling=true, density_unit="in", x_density=300, y_density=300, xmp_metadata="", name=nothing)
    local desc
    with_op_name(name, "EncodeJpeg") do
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

"""
    decode_png(contents; channels=0, dtype=UInt8, name="DecodePng")

Decode a PNG-encoded image to a `uint8` or `uint16` `Tensor`.

Args:
* `contents`: A 0D `Tensor` of type `string` containing the name of the image to decode.
* `channels`: An optional argument specifying how many color channels to use. `0` (default) means use the number of color channels in the decoded image. `1` means grayscale. `3` means RGB.
* `dtype`: Datatype of the output `Tensor`. Defaults to `UInt8`.
* `name`: An optional name for the operation.

Returns:

A `dtype` `Tensor` containing the decoded image, of size `[height, width, channels]`.
"""
function decode_png(contents; channels=0, dtype=UInt8, name=nothing)
    local desc
    with_op_name(name, "DecodePng") do
        desc = NodeDescription("DecodePng")
        add_input(desc, contents)
        desc["channels"] = Int64(channels)
        desc["dtype"] = dtype
    end
    Tensor(Operation(desc), 1)
end

"""
    encode_png(image; compression::Integer=-1, name="EncodePng")

Encode `uint8` or `uint16` `Tensor` `image` as a PNG. `image` has size `[height, width, channels]`.

Args:
* `contents`: A 0D `Tensor` of type `string` containing the name of the image to decode.
* `compression`: An optional `Integer` argument specifying what ZLIB compression level to use. Default is `-1` for encoder default. Higher numbers mean more compression, but are slower.
* `name`: An optional name for the operation.

Returns:

A zero dimensional `string` `Tensor` containing the PNG image name.
"""
function encode_png(image; compression::Integer=-1, name=nothing)
    local desc
    with_op_name(name, "EncodePng") do
        desc = NodeDescription("EncodePng")
        add_input(desc, image)
        desc["compression"] = Int64(compression)
    end
    Tensor(Operation(desc), 1)
end


@enum ResizeMethod BILINEAR NEAREST_NEIGHBOR BICUBIC AREA

"""
Resize `images` to `size` using the specified `method`.

Args:
*  `images`: Four dimensional `Tensor` of size `[batch_size, height, width, channels]` or three dimensional `Tensor` of size `[height, width, channels]`.
*  `new_height`: The new height for all the `images`.
*  `new_width`: The new width for all the `images`.
*  `method`: Optional argument specifying which interpolation method to use. Valid options are `BILINEAR` (default), `NEAREST_NEIGHBOR`, `BICUBIC`, and `AREA`.
*  `align_corners`: Optional `Bool` argument (default `false`), which forces all four corners of the resized `images` to be exactly aligned.

Raises:
*  `ValueError`: if the shape of `images` is incompatible with `new_width` and `new_height`.
*  `ValueError`: if the `type` or values of `new_width`/`new_height` are invalid.
*  `ValueError`: if an invalid `method` is given.

Returns:
A `Tensor` of the resized `images`.
"""
function resize_images(images, new_height, new_width; method=BILINEAR, align_corners=false, name=nothing)
    op_names = Dict(BILINEAR=>"ResizeBilinear", BICUBIC=>"ResizeBicubic")
    local desc
    with_op_name(name, "ResizeImages") do
        desc = NodeDescription(op_names[method])
        add_input(desc, images)
        dims = pack([convert_number(Int32,new_height), convert_number(Int32,new_width)])
        add_input(desc, dims)
        desc["align_corners"] = align_corners
    end
    Tensor(Operation(desc), 1)
end

"""
Crop the central region of the `image`.

Remove the outer parts of an `image` but retain the central region of the `image` along each dimension.

Args:
* `image`: A three dimensional `Float` `Tensor` to crop.
* `crop_fraction`: A float between `0` and `1` indicating what fraction of the image to retain.

Raises:
* `ValueError`: if `crop_fraction` is not between `0` and `1`.

Returns:

A 3D `Float` `Tensor` of the cropped `image`.
"""
function central_crop(image, crop_fraction; name=nothing)
    local desc
    with_op_name(name, "CentralCrop") do
        desc = NodeDescription("CentralCrop")
        add_input(desc, image)
        add_input(desc, Float32(crop_fraction))
    end
    Tensor(Operation(desc), 1)
end

function flip_up_down(image; name=nothing)
    local desc
    with_op_name(name, "FlipUpDown") do
        desc = NodeDescription("Reverse")
        add_input(desc, image)
        dims = Tensor([true, false, false])
        add_input(desc, dims)
    end
    Tensor(Operation(desc), 1)
end

function flip_left_right(image; name=nothing)
    local desc
    with_op_name(name, "FlipLeftRight") do
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
    @eval function $jl_func_name(image; name=nothing)
        local desc
        with_op_name(name, $tf_func_name) do
            desc = NodeDescription($tf_func_name)
            add_input(desc, image)
        end
        Tensor(Operation(desc), 1)
    end
end

end
