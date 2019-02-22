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

@back_for(Ops.add, function f(grad, x, y; kwargs...)
    return [constant(1.0), constant(1.0)] .*grad
end)

@back_for(Ops.sub, function f(grad, x, y; kwargs...)
    return [constant(1.0), constant(-1.0)] .*grad
end)

@back_for(Ops.neg, function f(grad, x; kwargs...)
    return constant(-1.0) .* grad
end)

function with_no_grad(f)
    old_tape = tape
    global tape = nothing
    res = f()
    global tape = old_tape
    return res
end

@back_for(Ops.exp, function f(grad, x; kwargs...)
    Ops.exp(x) .* grad
end)

@back_for(Ops.mean, function f(grad, x, reduction_indices; keep_dims=nothing, kwargs...)
    # assume reduction_indices is everything for now
    n_elem = float(num_elements(x))
    [grad .* Ops.fill(size(x), 1/constant(n_elem)), nothing]
end)

@back_for(Ops.sum, function f(grad, x, reduction_indices; keep_dims=nothing, kwargs...)
    # assume reduction_indices is everything for now
    [grad .* Ops.fill(size(x), constant(1.0)), nothing]
end)

@back_for(Ops.mul, function f(grad, x, y; kwargs...)
    return [grad.*y, grad.*x]
end)

@back_for(Ops.cast, function f(grad, x;  kwargs...)
    return grad
end)

@back_for(Ops.log, function f(grad, x; kwargs...)
    return 1/x .* grad
end)

@back_for(Ops.sin, function f(grad, x; kwargs...)
    return cos(x) .* grad
end)

@back_for(Ops.cos, function f(grad, x; kwargs...)
    return sin(x) .* grad
end)

@back_for(Ops.relu, function f(grad, x; kwarg...)
    # todo use relu grad
    ((x > 0) .* x) .* grad
end)

@back_for(Ops.mat_mul, function f(grad, x, y; transpose_a=nothing, transpose_b=nothing, kwargs...)
    # todo pay attension to transpose arguments
    grad_x = Ops.mat_mul(grad, y, transpose_b=true)
    grad_y = Ops.mat_mul(x, grad, transpose_a=true)
    return [grad_x, grad_y]
end)

@back_for(Ops.tanh, function f(grad, x; kwargs...)
    Ops.tanh_grad(x, grad)
end)

@back_for(Ops.sigmoid, function f(grad, x; kwargs...)
    Ops.sigmoid_grad(x, grad)
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
        back_op(out_grad, node.args...; node.kwargs...)
    end
    arg_grads = ensure_vector(arg_grads)
    for (i, arg) in enumerate(node.args)
        arg_grads[i] === nothing && continue
        grads[arg] = arg_grads[i]
        _grad(tape, arg, grads[arg], grads)
    end

    return
end

function grad(tape, tensor, in_tensors::AbstractArray, out_grad=constant(1.0))
    grads = Dict()
    _grad(tape, tensor, out_grad, grads)
    return [grads[tensor] for tensor in in_tensors]
end

function grad(tape, tensor, in_tensor, out_grad=constant(1.0))
    grad(tape, tensor, [in_tensor], out_grad)[1]
end
