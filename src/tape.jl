using MacroTools
import MacroTools: splitdef, combinedef

mutable struct TapeNode
    op
    args
    kwargs
end


TapeNode(op, args; kwargs...) =  TapeNode(op, args, kwargs)

mutable struct Tape
    nodes::Dict{TensorHandle, TapeNode}
end

Tape() = Tape(Dict{TensorHandle, TapeNode}())

tape = nothing

function set_tape(new_tape=nothing)
    if new_tape === nothing
        new_tape = Tape()
    end
    global tape = new_tape
    return tape
end

function add_node(t, node)
    tape === nothing && return
    tape.nodes[t] = node
end

grad_fns = Dict()

macro back_for(target, fn)
    def = splitdef(fn)
    if def[:name] == :f
        def[:name] = Symbol(string(target, "_", "backwards"))
    end
    quote
        $(esc(combinedef(def)))
        grad_fns[$target] = $(def[:name])
    end
end

@back_for(Ops.add, function f(x, y; kwargs...)
    return [constant(1.0), constant(1.0)]
end)

@back_for(Ops.sub, function f(x, y; kwargs...)
    return [constant(1.0), constant(-1.0)]
end)

@back_for(Ops.neg, function f(x; kwargs...)
    return constant(-1.0)
end)

function with_no_grad(f)
    old_tape = tape
    global tape = nothing
    res = f()
    global tape = old_tape
    return res
end

@back_for(Ops.exp, function f(x; kwargs...)
    Ops.exp(x)
end)

@back_for(Ops.mean, function f(x, reduction_indices; keep_dims=nothing, kwargs...)
    # assume reduction_indices is everything for now
    n_elem = float(num_elements(x))
    [Ops.fill(size(x), 1/constant(n_elem)), nothing]
end)

@back_for(Ops.sum, function f(x, reduction_indices; keep_dims=nothing, kwargs...)
    # assume reduction_indices is everything for now
    [Ops.fill(size(x), constant(1.0)), nothing]
end)


@back_for(Ops.mul, function f(x, y; kwargs...)
    return [y, x]
end)

@back_for(Ops.cast, function f(x;  kwargs...)
    return constant(1.0)
end)


@back_for(Ops.log, function f(x; kwargs...)
    return 1/x
end)

@back_for(Ops.sin, function f(x; kwargs...)
    return cos(x)
end)

@back_for(Ops.cos, function f(x; kwargs...)
    return sin(x)
end)

@back_for(Ops.relu, function f(x; kwarg...)
    (x > 0) .* x
end)


ensure_vector(x::AbstractArray) = x
ensure_vector(x) = [x]

function _grad(tape::Tape, tensor, out_grad, grads)
    if !haskey(tape.nodes, tensor)
        return
    end
    
    node = tape.nodes[tensor]
    back_op = grad_fns[node.op]
    arg_grads = with_no_grad() do
        back_op(node.args...; node.kwargs...)
    end
    arg_grads = ensure_vector(arg_grads)
    for (i, arg) in enumerate(node.args)
        arg_grads[i] === nothing && continue
        grads[arg] = arg_grads[i].*out_grad
        _grad(tape, arg, grads[arg], grads)
    end

    return
end

function grad(tape, tensor, in_tensors::AbstractArray, out_grad=1.0)
    grads = Dict()
    _grad(tape, tensor, out_grad, grads)
    return [grads[tensor] for tensor in in_tensors]
end

function grad(tape, tensor, in_tensor, out_grad=1.0)
    grad(tape, tensor, [in_tensor], out_grad)[1]
end
