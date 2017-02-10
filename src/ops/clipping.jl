function clip_by_value(t, clip_value_min, clip_value_max; name=nothing)
    local out
    with_op_name(name, "ClipByValue") do
        out = max(min(t, clip_value_max), clip_value_min)
    end
    out
end

Base.clamp(t::AbstractTensor, min_value, max_value) = clip_by_value(t, min_value, max_value)

function clip_by_norm(t, clip_norm; axes=nothing, name=nothing)
    local out
    t = Tensor(t)
    with_op_name(name, "ClipByNorm") do
        clip_norm = cast(Tensor(clip_norm), eltype(t))
        norm = sqrt(reduce_sum(t.*t, reduction_indices=axes))
        factor = min(clip_norm/norm, cast(constant(1), eltype(t)))
        out = t .* factor
    end
    out
end

function clip_by_norm(t::IndexedSlices, clip_norm)
    IndexedSlices(clip_by_norm(t.values, clip_norm), t.indices)
end

@not_implemented function clip_by_average_norm(t, clip_norm; name="")
end

function clip_by_global_norm(t_list, clip_norm; use_norm=nothing, name=nothing)
    local out, gn
    if isempty(t_list)
        error("Must pass at least one tensor to clip_by_global_norm")
    end
    clip_tensor(t, ratio) = t .* ratio
    clip_tensor(t::IndexedSlices, ratio) = IndexedSlices(t.values .* ratio, t.indices)
    with_op_name(name, "ClipByGlobalNorm") do
        clip_norm = cast(Tensor(clip_norm), eltype(t_list[1]))
        if use_norm === nothing
            gn = global_norm(t_list)
        else
            gn = use_norm
        end
        out = [clip_tensor(t, clip_norm / max(gn, clip_norm)) for t in t_list]
    end
    [out, gn]
end

function global_norm(t_list; name=nothing)
    local out
    tensor_value(t) = Tensor(t)
    tensor_value(t::IndexedSlices) = t.values
    with_op_name(name, "GlobalNorm") do
        out = sqrt(add_n([reduce_sum(tensor_value(t).^2) for t in t_list]))
    end
    out
end
