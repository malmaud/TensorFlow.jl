for (func, name) in [
    (:is_finite, "IsFinite"),
    (:is_nan, "IsNan"),
    (:is_inf, "IsInf")]
    @eval function $func(t::AbstractTensor; name=$name)
        local desc
        with_op_name(name) do
            desc = NodeDescription($name)
            add_input(desc, Tensor(t))
        end
        Tensor(Operation(desc), 1)
    end
end
