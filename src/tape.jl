using MacroTools

mutable struct Tensor
    x
end

mutable struct TapeNode
    op
    args
end

import Base: *, log, sin

mutable struct Tape
    nodes::Dict{Tensor, TapeNode}
end

Tape() = Tape(Dict{Tensor, TapeNode}())

tape = nothing

function set_tape(new_tape=nothing)
    if new_tape === nothing
        new_tape = Tape()
    end
    global tape = new_tape
    return tape
end

function add_node(t, node)
    tape.nodes[t] = node
end

function *(t1::Tensor, t2::Tensor)
    t3 = Tensor(t1.x*t2.x)
    node = TapeNode(*, [t1, t2])
    add_node(t3, node)
    return t3
end

function *(t1::Tensor, t2::AbstractFloat)
    return Tensor(t1.x*t2)    
end

grad_fns = Dict()

macro back_for(target, fn)
    def = splitdef(fn)
    quote
        $(esc(fn))
        grad_fns[$target] = $(def[:name])
    end
end


@back_for(*, function mul_backwards(args)
    return [args[2], args[1]]
end)


@back_for(log, function log_backwards(args)
    return [Tensor(1/args[1].x)]
end)

function Base.sin(t::Tensor)
end

@back_for(sin, function sin_backwards(args)
    return [Tensor(cos(args[1].x))]
end)

function log(t::Tensor)
    res = Tensor(log(t.x))
    node = TapeNode(log, [t])
    add_node(res, node)
    return res
end

function grad(tape::Tape, tensor, out_grad, grads)
    if !haskey(tape.nodes, tensor)
        return
    end
    
    node = tape.nodes[tensor]
    back_op = grad_fns[node.op]
    arg_grads = back_op(node.args)

    
    for (i, arg) in enumerate(node.args)
        grads[arg] = arg_grads[i]*out_grad
        grad(tape, arg, grads[arg].x, grads)
    end

    return
end

function grad(tape, tensor, out_grad=1.0)
    grads = Dict()
    grad(tape, tensor, out_grad, grads)
    return grads
end
