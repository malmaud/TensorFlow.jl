module image

import ..TensorFlow: NodeDescription, get_def_graph, get_name, add_input, Node

function decode_jpeg(contents; channels=0, ratio=1, fancy_upscaling=true, try_recover_truncated=false, acceptable_fraction=1.0, name="")
    desc = NodeDescription(get_def_graph(), "DecodeJpeg", get_name(name))
    add_input(desc, contents)
    desc["acceptable_fraction"] = Float32(acceptable_fraction)
    desc["channels"] = Int64(channels)
    desc["fancy_upscaling"] = fancy_upscaling
    desc["ratio"] = Int64(ratio)
    desc["try_recover_truncated"] = try_recover_truncated
    Node(desc)
end

end
