using MacroTools

# Use Holy traits to make if something is a known Op or not
abstract OpRegistration
immutable RegisteredOp<:OpRegistration end
immutable NotRegisteredOp<:OpRegistration end

is_registered_op(::DataType) = NotRegisteredOp() # By default nothing is registered


macro op(f)
    f = longdef(f) #convert to long form function
    opname = f.args[1].args[1]
    @assert(isdefined(:tf)) # Need tf as name for module where this code is located
    quote
        @Base.__doc__ $f

        tf.is_registered_op(::Type{typeof($(opname))}) = tf.RegisteredOp()
        # Mark it as registered by giving its type the trait
        # Need  the `Type{...}` so that both DataTypes constructors (eg FIFOQueue), and functions work

        $(opname)
    end |> esc
end


# How to insert the  name into functions etc
# This function takes in a function and its posible name
# and returns a new function that will call to the orginal
# with the function name inserted as appropriate


withname(::typeof(get_variable), name) = (args...; kwargs...) -> begin
    if typeof(args[1])<:AbstractString
        get_variable(args...; kwargs...)
    else # No name provided
        get_variable(name, args...; kwargs...)
    end
end

withname{F}(f::F, name) = withname(is_registered_op(F), f, name) # Should be at compile time converted to one of the two below
withname(::NotRegisteredOp, f, name) = (args...; kws...) -> f(args...; kws...)
withname(::RegisteredOp, f, name) = (args...; kws...) -> begin
    if !any(first.(kws) .== :name) #name is not already there
        push!(kws, (:name, name))
    end
    f(args...; kws...)
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
    while_expr = Expr(:call, :(TensorFlow.while_loop))
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
    if ex.head == :while
        tf_while(ex)
    else
        MacroTools.prewalk(ex) do x
            if @capture(x, X_ = f_(args__))
                :($X = withname($f, $(string(X)))($(args...)))
            else
                x
            end
        end
    end |> esc
end

#=
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
=#
