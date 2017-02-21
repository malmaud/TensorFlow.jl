function _tf(ex)
    @assert ex.head == Symbol("=")
    name = ex.args[1]
    call = copy(ex.args[2])
    @assert call.head == :call
    if length(call.args) >=2 && isa(call.args[2], Expr) && call.args[2].head == :parameters
        params = call.args[2]
    else
        params = Expr(:parameters)
        insert!(call.args, 2, params)
    end
    push!(params.args, Expr(:kw, :name, string(name)))
    quote
        $name = $call
    end
end

"""
    @tf

Automatically name a tensor by the name of the variable it is assigned to.

For example,
`@tf i = constant(1)` creates a node with name "i", exactly as if you
wrote `i = constant(1, name="i")`.

Can also be applied to a block of assignments:
```
@tf begin
  i = constant(1)
  j = constant(2)
end
```
"""
macro tf(ex)
    is_assign(arg::Expr) = arg.head == Symbol("=")
    is_assign(arg) = false
    if ex.head == :block
        res = Expr(:block)
        for arg in ex.args
            if is_assign(arg)
                push!(res.args, _tf(arg))
            else
                push!(res.args, arg)
            end
        end
    else
        res = _tf(ex)
    end
    esc(res)
end
