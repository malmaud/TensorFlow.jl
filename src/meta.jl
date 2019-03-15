using MacroTools
using Compat

const tf = TensorFlow

# Use Holy traits to define if something is a known Op or not
abstract type OpRegistration end
struct RegisteredOp <: OpRegistration end
struct NotRegisteredOp <: OpRegistration end

# By default nothing is registered
is_registered_op(args...) =  TensorFlow.NotRegisteredOp()


macro op(f)
    @capture(f,
        function opname_(args__; kwargs__) body_ end |
        (opname_(args__; kwargs__) = body_) |
        function opname_(args__) body_ end  |
        (opname_(args__) = body_)
    ) || error("Invalid usage of @op")
    

    @assert(@isdefined(tf)) # Need tf as name for module where this code is located

    # TODO Refactor this a bit
    register_block = quote
        # Mark it as registered by giving its type, and sig the trait
        if isa($(opname), Function)
            tf.is_registered_op(::Type{typeof($(opname))}, $(args...)) = tf.RegisteredOp()
        elseif isa($(opname), DataType)
            tf.is_registered_op(::Type{$(opname)}, $(args...)) = tf.RegisteredOp()
        else
            @warn("@op used on " * string($(opname)) * " which does not seem to be a suitable type for an operation.")
        end
    end

    res = quote
        @Base.__doc__ $f
        $register_block
        #$(opname)
    end |> esc
    res
end


# How to insert the  name into functions etc
# This function takes in a function and its posible name
# and returns a new function that will call to the orginal
# with the function name inserted as appropriate

function get_variable end # pre-declare.

function withname(::typeof(get_variable), name, args...; kwargs...)
    if length(args) ≥ 1 && isa(args[1], AbstractString)
        get_variable(args...; kwargs...)
    else # No name provided
        get_variable(name, args...; kwargs...)
    end
end

function withname(d::Type{T}, name, args...; kwargs...) where T
    withname(is_registered_op(d, args...), d, name, args...; kwargs...)
end
function withname(f::F, name, args...; kwargs...) where F<:Function
    withname(is_registered_op(F, args...), f, name, args...; kwargs...)
end

withname(::NotRegisteredOp, f, name, args...; kws...) = f(args...; kws...)
function withname(::RegisteredOp, f, name, args...; kws...)
    if !any(keys(kws) .== :name) # name is not already there
        kws = (kws..., name=name)
    end
    f(args...; kws...)
end

function tf_while(ex)
    (@capture ex begin
        while cond_
            block_
        end
    end) || error("tf_while expects a `while` loop")

    return_val = block.args[end]
    loop_err() = error("loop must end with a list of pairs")
    (@capture return_val [return_items__]) || loop_err()

    # derive the variables involved from the last expression
    vars = []
    for item in return_items
        (@capture item (var_name_=>var_value_)) || loop_err()
        push!(vars, (var_name=>var_value))
    end

    while_expr = :(TensorFlow.while_loop())

    loop_func = Expr(:->, Expr(:tuple, [var[1] for var in vars]...))

    # condition argument in TensorFlow.while_loop
    cond_func = deepcopy(loop_func)
    push!(cond_func.args, cond)
    push!(while_expr.args, cond_func)

    # body argument in TensorFlow.while_loop
    iter_func = deepcopy(loop_func)
    block_part = :(begin end)
    for arg in block.args[1:end-1]
        arg.head == :line && continue
        push!(block_part.args, arg)
    end
    block_return_part = :([$((var[2] for var in vars)...)])
    push!(block_part.args, block_return_part)
    push!(iter_func.args, block_part)
    push!(while_expr.args, iter_func)

    # variables argument in TensorFlow.while_loop
    var_list = :([$((var[1] for var in vars)...)])
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
        TensorFlow.tf_while(ex)
    else
        # Recursively search the expression, looking for assignments of function calls
        # If they are found replace them with `withname` wrapped calls
        # and then search with in them
        MacroTools.prewalk(ex) do x
            if @capture(x, X_ = f_(args__; kwargs__)) # semicolon breaks it
                :($X = TensorFlow.withname($f, $(string(X)), $(args...); $(kwargs...)))
            elseif @capture(x, X_ = f_(args__))
                :($X = TensorFlow.withname($f, $(string(X)), $(args...)))
            else
                x
            end
        end
    end |> esc
end
