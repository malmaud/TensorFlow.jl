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

@not_implemented function merge_summary()
end

@not_implemented function merge_all_summaries()
end

@not_implemented function image_summary()
end
