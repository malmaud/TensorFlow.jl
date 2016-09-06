function scalar_summary(tags, values; collections=[:Summaries], name="")
    desc = NodeDescription("ScalarSummary", get_name(name))
    add_input(desc, Tensor(tags))
    add_input(desc, Tensor(values))
    t = Tensor(Operation(desc))
    for collection in collections
        add_to_collection(collection, t)
    end
    return t
end

@not_implemented function audio_summary()
end

function histogram_summary(tag, values; collections=[:Summaries], name="")
    desc = NodeDescription("HistogramSummary", get_name(name))
    add_input(desc, Tensor(tag))
    add_input(desc, Tensor(values))
    t = Tensor(Operation(desc))
    foreach(c->add_to_collection(c, t), collections)
    t
end

function merge_summary(inputs; collections=[:Summaries], name="")
    desc = NodeDescription("MergeSummary", get_name(name))
    add_input(desc, inputs)
    t = Tensor(Operation(desc))
    for collection in collections
        add_to_collection(collection, t)
    end
    return t
end

function merge_all_summaries(key=:Summaries)
    merge_summary(get_collection(:Summaries), collections=[])
end

function image_summary(tag, tensor; max_images=3, collections=[:Summaries], name="")
    desc = NodeDescription("ImageSummary")
    add_input(desc, tag)
    add_input(desc, tensor)
    desc["max_images"] = Int64(max_images)
    
end
