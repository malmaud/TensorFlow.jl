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

@not_implemented function histogram_summary()
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

@not_implemented function image_summary()
end
