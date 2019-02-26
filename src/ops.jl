using MacroTools
import Base: log, exp, +, -, *, /, ^, sin, cos, tan, asin, acos, atan, div, tanh, sqrt, abs, floor, ceil, floor, sign
import TensorFlow
const tf = TensorFlow # so know where op_funcs is defined

"""
    tf_promote(args...)

Converts all arguments to `Tensor`s whose element type is sufficiently wide
to losslessly represent all arguments. Analogous to `promote`, but for tensors.

Argumens that are already tensors are dynamically casted to tensors of the
appropriate types using `TensorFlow.cast`.. Non-tensor argments are converted
to tensors with `constant`.
"""
function tf_promote(args...)
    big_type = eltype(args[1])
    for arg in args[2:end]
        big_type = promote_type(big_type, eltype(arg))
    end
    new_args = []
    for arg in args
        if isa(arg, AbstractArray)
            push!(new_args, arg)
        else
            if in_eager_mode()
                push!(new_args, Ops.cast(arg, DstT = big_type))  # TODO implement promotion
            else
                push!(new_args, convert(Tensor{big_type}, arg))
            end
        end
    end
    (new_args...,)
end

macro define_binary(jl_func, tf_func)
    quote
        @op $jl_func(t1::AbstractTensor, t2::AbstractTensor; kwargs...) = $tf_func(tf_promote(t1, t2)...; kwargs...)
        @op $jl_func(t1::AbstractTensor, t2; kwargs...) = $jl_func(t1, Tensor(t2); kwargs...)
        @op $jl_func(t1, t2::AbstractTensor; kwargs...) = $jl_func(Tensor(t1), t2; kwargs...)
    end |> esc
end

macro define_unary(jl_func, tf_func)
    quote
        @Base.__doc__ @op function $jl_func(t::AbstractTensor; kwargs...)
            $tf_func(t; kwargs...)
        end
    end |> esc
end

macro define_broadcast_unary(jl_op)
    quote
        Broadcast.broadcasted(::typeof($jl_op), t::AbstractTensor) = $jl_op(t)
    end
end

macro define_broadcast_binary(jl_op)
    quote
        Broadcast.broadcasted(::typeof($jl_op), t1::AbstractTensor, t2::AbstractTensor) = $jl_op(t1, t2)
        Broadcast.broadcasted(::typeof($jl_op), t1::AbstractTensor, t2) = $jl_op(t1, t2)
        Broadcast.broadcasted(::typeof($jl_op), t1, t2::AbstractTensor) = $jl_op(t1, t2)
    end
end

# TODO can this be subsumed by 'define_broadcast_binary'?
macro define_broadcast(jl_op, tf_func)
    quote
        Base.Broadcast.broadcasted(::typeof($jl_op), t1::AbstractTensor, t2::AbstractTensor) = $tf_func(tf_promote(t1, t2)...)
        Base.Broadcast.broadcasted(::typeof($jl_op), t1::AbstractTensor, t2) = $tf_func(tf_promote(t1, Tensor(t2))...)  # TODO don't replicate the tf_promote calls
        Base.Broadcast.broadcasted(::typeof($jl_op), t1, t2::AbstractTensor) = $tf_func(tf_promote(Tensor(t1), t2)...)
        Base.Broadcast.broadcasted(::typeof($jl_op), t1::AbstractTensor, t2::Base.Broadcast.Broadcasted) = $tf_func(tf_promote(t1, Tensor(collect(t2)))...)
        Base.Broadcast.broadcasted(::typeof($jl_op), t1::Base.Broadcast.Broadcasted, t2::AbstractTensor) = $tf_func(tf_promote(Tensor(collect(t1)), t2)...)
    end |> esc
end

macro not_implemented(ff)
    if @capture(ff, function name_(args__) body__ end)
        quote
            function $(esc(name))(args...)
                error("Not implemented yet")
            end
        end
    else
        error("Invalid use of not_implemented")
    end
end

function tfimport(expr)
    res = @capture expr begin
        fname_(args__)
    end
    if res
        quote
            import_op($(string(fname)))($([esc(arg) for arg in args]...))
        end
    else
        res = @capture expr begin
            fname_
        end
        if res
            jlname = opname_to_jlname(string(fname))
            quote
                $(esc(jlname)) = import_op($(string(fname)))
            end
        else
            error("Invalid use of @tfimport on $(expr)")
        end
    end
end

macro tfimport(expr)
    tfimport(expr)
end

function capitalize(s)
    string(uppercase(s[1]), s[2:end])
end

capitalize(s::Symbol) = capitalize(string(s))

function get_name(name = "node")
    graph = get_def_graph()
    name_idx = graph.name_idx
    if name == ""
        name = "node"
    end
    cur_idx = get(name_idx, name, 1)
    name_idx[name] = cur_idx + 1
    if cur_idx == 1
        name
    else
        string(name, "_", cur_idx)
    end
end

"""
Inserts a placeholder for a tensor that will be always fed.

**Important**: This tensor will produce an error if evaluated. Its value must
be fed using the third argument argument to `run(::Session, ...`.

For example:

```julia
x = placeholder(Float32, shape=[1024, 1024])
y = x*x
sess = Session()
print(run(sess, y))  # ERROR: will fail because x was not fed.

rand_array = rand(1024, 1024)
print(run(sess, y, Dict(x=>rand_array)))  # Will succeed.
```

Args:
  * dtype: The type of elements in the tensor to be fed.
  * shape: The shape of the tensor to be fed (optional). If the shape is not
    specified, you can feed a tensor of any shape.
  * name: A name for the operation (optional).

Returns:
  A `Tensor` that may be used as a handle for feeding a value, but not
  evaluated directly.
"""
@op function placeholder(dtype; name = nothing, shape = nothing)
    local node
    with_op_name(name, "placeholder") do
        graph = get_def_graph()
        desc = NodeDescription("Placeholder")
        desc["dtype"] = dtype
        node = Operation(desc)
        node_name = get_cur_node_name()
    end
    tensor = Tensor(node, 1)
    if shape !== nothing
        TensorFlow.set_tensor_shape(tensor, shape)
    end
    return tensor
end



"""
    struct_map(operation, args...)

Run `operation` separately over the deconstructed version of each arg in args,
then reconstruct the output to be of the same struct type as the args
(eg, LSTMStateTuple).
"""
function struct_map(op, args...)
    @assert !isempty(args)
    tensors = get_tensors.(args)
    N = length(tensors[1])
    for i in eachindex(tensors)
        @assert length(tensors[i]) == N
    end
    mapped_tensors = []
    for n in 1:N
        tensor_list = []
        for tensor in tensors
            push!(tensor_list, tensor[n])
        end
        mapped_tensor = op(tensor_list...)
        push!(mapped_tensors, mapped_tensor)
    end
    build_output(args[1], mapped_tensors)
end


include("ops/imported_ops.jl")
include("ops/math.jl")
include("ops/sequences.jl")
include("ops/control_flow.jl")
include("ops/logical.jl")
include("ops/debugging.jl")
include("ops/comparison.jl")
include("ops/transformations.jl")
include("ops/nn.jl")
include("ops/image.jl")
include("ops/queues.jl")
include("ops/clipping.jl")
include("ops/init_ops.jl")
