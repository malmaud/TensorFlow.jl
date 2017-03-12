function _tf(ex)
    @assert ex.head == Symbol("=")
    name = ex.args[1]
    call = copy(ex.args[2])
    @assert call.head == :call

    # Special-case :get_variable, which takes `name` as its first argument instead
    # of a keyword argument
    if call.args[1] == :get_variable
        insert!(call.args, 2, string(name))
    else
        if length(call.args) >=2 && isa(call.args[2], Expr) && call.args[2].head == :parameters
            params = call.args[2]
        else
            params = Expr(:parameters)
            insert!(call.args, 2, params)
        end
        push!(params.args, Expr(:kw, :name, string(name)))
    end
    quote
        $name = $call
    end
end

function tf_while(ex)
    ex.head == :while || error("tf_while expects a `while` loop")
    cond = ex.args[1]
    block = ex.args[2]
    return_val = block.args[end]
    @assert return_val.head == :vect
    vars = []
    for item in return_val.args
        @assert item.head == Symbol("=>")
        var_name = item.args[1]
        var_value = item.args[2]
        push!(vars, (var_name=>var_value))
    end
    while_expr = Expr(:call, :while_loop)
    loop_func = Expr(:->, Expr(:tuple, [var[1] for var in vars]...))
    cond_func = deepcopy(loop_func)
    push!(cond_func.args, cond)
    push!(while_expr.args, cond_func)
    iter_func = deepcopy(loop_func)
    vec_part = Expr(:vect, [var[2] for var in vars]...)
    block_part = Expr(:block)
    for arg in block.args[1:end-1]
        arg.head == :line && continue
        push!(block_part.args, arg)
    end
    push!(block_part.args, vec_part)
    push!(iter_func.args, block_part)
    push!(while_expr.args, iter_func)
    var_list = Expr(:vect, [var[1] for var in vars]...)
    push!(while_expr.args, var_list)
    while_expr
end

"""
    @tf

When applied to assignment, automatically name a tensor by the name of the variable it is assigned to.

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

When applied to a `while` loops, automatically transform the loop to a TensorFlow
`while_loop`.

For example,

```
i = constant(0)
loop_sum = constant(0)
@tf while i<10
  [i=>i+1, loop_sum=>loop_sum+i]
end
```

becomes `while_loop((i,loop_sum)->i<10, (i,loop_sum)->[i+1, loop_sum+i], [i, loop_sum])`.
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
    elseif ex.head == Symbol("=")
        res = _tf(ex)
    elseif ex.head == :while
        res = tf_while(ex)
    else
        warn("@tf macro had no effect")
        res = ex
    end
    esc(res)
end
