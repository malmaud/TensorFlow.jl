module image

export
decode_jpeg,
decode_png,
resize_images

import ..TensorFlow: NodeDescription, get_def_graph, get_name, add_input, Operation, pack, convert_number, AbstractOperation, Tensor

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

end
