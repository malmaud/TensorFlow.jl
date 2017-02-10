for (func, name) in [
    (:is_finite, "IsFinite"),
    (:is_nan, "IsNan"),
    (:is_inf, "IsInf")]
    @eval function $func(t::AbstractTensor; name=nothing)
        local desc
        with_op_name(name, $name) do
            desc = NodeDescription($name)
            add_input(desc, Tensor(t))
        end
        Tensor(Operation(desc), 1)
    end
end

"""
    Print(input, data; message=nothing, first_n=nothing, summarize=nothing, name=nothing)

Prints a list of tensors.

This is an identity op with the side effect of printing `data` when
evaluating.

Args:
  input_: A tensor passed through this op.
  data: A list of tensors to print out when op is evaluated.
  message: A string, prefix of the error message.
  first_n: Only log `first_n` number of times. Negative numbers log always;
           this is the default.
  summarize: Only print this many entries of each tensor. If None, then a
             maximum of 3 elements are printed per input tensor.
  name: A name for the operation (optional).

Returns:
  Same tensor as `input_`.
"""
function Print(input, data; message=nothing, first_n=nothing, summarize=nothing, name=nothing)
    local desc
    with_op_name(name, "Print") do
        desc = NodeDescription("Print")
        add_input(desc, input)
        add_input(desc, data)
        if message !== nothing
            desc["message"] = message
        end
        if first_n !== nothing
            desc["first_n"] = Int64(first_n)
        end
        if summarize !== nothing
            desc["summarize"] = Int64(summarize)
        end
    end
    Tensor(Operation(desc), 1)
end
