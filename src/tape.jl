using MacroTools
import MacroTools: splitdef, combinedef

struct TapeNode
    op::Function
    args::Vector
    results::Vector
    kwargs::Dict
end

TapeNode(op, args, results; kwargs...) =  TapeNode(op, args, results, kwargs)

mutable struct Tape
    nodes::Dict{EagerTensor, TapeNode}
    attrs::Dict
end

Tape(;kwargs...) = Tape(Dict{EagerTensor, TapeNode}(), Dict(kwargs...))

struct TapeContext <: Context
    tape::Union{Tape, Nothing}
end

create_tape() = set_tape(Tape())

function set_tape(new_tape)
    push!(global_context, TapeContext(new_tape))
    return new_tape
end

function with_tape(block, tape=Tape())
    ctx = TapeContext(tape)
    with_context(block, ctx)
end

function get_tape()
    tape_context = context_value(TapeContext)
    if tape_context === nothing
        return nothing
    else
        return tape_context.tape
    end
end

function add_node(t, node)
    tape = get_tape()
    tape === nothing && return
    tape.nodes[t] = node
end

function backwards
end

macro back_for(target, fn)
    def = splitdef(fn)
    if def[:name] == :f
        def[:name] = Symbol(string(target, "_", "backwards"))
    end
    backwards_expr = :(backwards(::typeof($target)) = $(def[:name]))
    quote
        $(esc(combinedef(def)))
        $(esc(backwards_expr))
    end
end

@back_for(Ops.add, function f(grad, x, y; kwargs...)
    println("Add got $grad, $x, $y")
    return [constant(1.0).*grad, constant(1.0).*grad]
end)

@back_for(Ops.sub, function f(grad, x, y; kwargs...)
    return [constant(1.0).*grad, constant(-1.0).*grad]
end)

@back_for(Ops.neg, function f(grad, x; kwargs...)
    return constant(-1.0) .* grad
end)

@back_for(Ops.pow, function f(grad, x, y; kwargs...)
    [y.* (x.^(y.-1)), nothing]
end)

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
    Ops.relu_grad(grad, x)
end)

@back_for(Ops.mat_mul, function f(grad, x, y; transpose_a=nothing, transpose_b=nothing, kwargs...)
    # todo pay attension to transpose arguments
    grad_x = Ops.mat_mul(grad, y, transpose_b=true)
    grad_y = Ops.mat_mul(x, grad, transpose_a=true)
    return [grad_x, grad_y]
end)

@back_for(Ops.tanh, function f(grad, x; output=nothing, kwargs...)
    Ops.tanh_grad(output[1], grad)
end)

@back_for(Ops.sigmoid, function f(grad, x; output=nothing, kwargs...)
    Ops.sigmoid_grad(output[1], grad)
end)

@back_for(Ops.sqrt, function f(grad, x; output=nothing, kwargs...)
    Ops.sqrt_grad(output[1], grad)
end)

@back_for(Ops.bias_add, function f(grad, x, y; kwargs...)
    [grad, Ops.bias_add_grad(grad)]
end)

function with_no_grad(f)
    res = with_context(f, TapeContext(nothing))
    return res
end

ensure_vector(x::AbstractArray) = x
ensure_vector(x) = [x]

function _grad(tape::Tape, tensor, out_grad, grads)
    if !haskey(tape.nodes, tensor)
        return
    end
    
    node = tape.nodes[tensor]
    back_op = backwards(node.op)
    arg_grads = with_no_grad() do
        back_op(out_grad, node.args...; output=node.results, node.kwargs...)
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
    get(tape.attrs, "preserve", false) || empty!(tape.nodes)
    return [get(grads, tensor, nothing) for tensor in in_tensors]
end

function grad(tape, tensor, in_tensor, out_grad=constant(1.0))
    grad(tape, tensor, [in_tensor], out_grad)[1]
end
