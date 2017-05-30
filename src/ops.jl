using MacroTools
import Base: log, exp, +, -, *, /, ^, sin, cos, tan, asin, acos, atan, div, tanh, sqrt, abs, floor, ceil, floor, sign
import TensorFlow
const tf = TensorFlow # so know where op_funcs is defined

if VERSION < v"0.6.0-dev.1632"
    import Base: .*, .+, ./, .-, .^, .==, .!=
end

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
            push!(new_args, convert(Tensor{big_type}, arg))
        end
    end
    (new_args...)
end

macro define_binary(jl_func, tf_func)
    quote
        $jl_func(t1::AbstractTensor, t2::AbstractTensor) = $tf_func(tf_promote(t1, t2)...)
        $jl_func(t1::AbstractTensor, t2) = $tf_func(t1, Tensor(t2))
        $jl_func(t1, t2::AbstractTensor) = $tf_func(Tensor(t1), t2)
    end |> esc
end

macro define_broadcast(jl_op, tf_func)
    quote
        Base.broadcast(::typeof($jl_op), t1::AbstractTensor, t2::AbstractTensor) = $tf_func(tf_promote(t1, t2)...)
        Base.broadcast(::typeof($jl_op), t1::AbstractTensor, t2) = $tf_func(t1, Tensor(t2))
        Base.broadcast(::typeof($jl_op), t1, t2::AbstractTensor) = $tf_func(Tensor(t1), t2)
    end |> esc
end

macro not_implemented(f)
    res = @match f begin
        function name_(args__)
            body__
        end => name, args
    end
    res === nothing && error("Invalid use of not_implemented")
    func_name, args = res
    quote
        function $(esc(func_name))(args...)
            error("Not implemented yet")
        end
    end
end

function capitalize(s)
    string(uppercase(s[1]), s[2:end])
end

capitalize(s::Symbol) = capitalize(string(s))

function get_name(name="node")
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
@op function placeholder(dtype; name=nothing, shape=nothing)
    local node
    with_op_name(name, "placeholder") do
        graph = get_def_graph()
        desc = NodeDescription("Placeholder")
        desc["dtype"] = dtype
        node = Operation(desc)
        if shape === nothing
            graph.shapes[get_cur_node_name()] = ShapeInference.TensorShape(nothing)
        else
            dims = Nullable{Int}[]
            for dim in shape
                if dim==-1 || dim==nothing
                    push!(dims, Nullable{Int}())
                else
                    push!(dims, Nullable(dim))
                end
            end
            graph.shapes[get_cur_node_name()] = ShapeInference.TensorShape(dims)
        end
    end
    Tensor(node, 1)
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

immutable OpFunc
    expr::Expr
    docstring::String
    name::Symbol
end

function keyword_escape(s)
    keywords = ["const", "type"]
    if (s ∈ keywords) || Base.isoperator(Symbol(s))
        s = string(s, "_")
    end
    s
end

"""
    opname_to_jlname(name)

Converts a TensorFlow operation name `name` into a Julia
function name. Eg UnsortedSegmentSum->unsorted_segment_sum.
"""
function opname_to_jlname(name)
    cur_word = Vector{Char}()
    words = Vector{String}()
    for idx in 1:length(name)
        cur_char = name[idx]
        push!(cur_word, cur_char)
        word_end = false
        if idx == length(name)
            word_end = true
        else
            next_char = name[idx+1]
            if idx < length(name)-1
                next_next_char = name[idx+2]
                if isupper(cur_char) && isupper(next_char) && islower(next_next_char)
                    word_end=true
                end
            end
            if islower(cur_char) && isupper(next_char)
                word_end = true
            end
        end
        if word_end
            push!(words, lowercase(join(cur_word)))
            empty!(cur_word)
        end
    end
    result = join(words, "_")
    escaped = keyword_escape(result)
    Symbol(escaped)
end

function is_internal_arg(arg)
    arg._type == "type" && ismatch(r"^T", arg.name)
end

function to_function(op::tensorflow.OpDef)
    jl_name = Symbol(lowercase(op.name))
    jl_name = opname_to_jlname(op.name)
    inputs = []
    input_block = quote end
    convert_block = quote end
    type_sets = Dict{String, Vector{Symbol}}()
    for (i, input) in enumerate(op.input_arg)
        sym = Symbol(:x, i)
        push!(inputs, sym)
        if !isempty(input.type_attr)
            type_set = get!(type_sets, input.type_attr, Symbol[])
            push!(type_set, sym)
        end
    end
    for (input_idx, input) in enumerate(op.input_arg)
        sym = inputs[input_idx]
        convert_target = tf.Tensor{Any}

        # Heuristic for when 1-based conversion is necessary
        # Generally, you can tell by the name of the type attribute.
        # One exception is split_dim, which has no type attribute but needs to 1-adjusted
        # Another is 'range', which uses 'Tidx' for the attribute name although no conversion should be done
        if (input.type_attr ∈ ["Index", "Tidx", "Tindices", "Tdim", "TI"] && jl_name != :range && input.name ∉ ["size", "shape"]) || input.name ∈ ["split_dim"]
            diff_expr = quote
                #converted = converted - 1
                $sym = $sym - convert(tf.Tensor{eltype($sym)}, 1)
            end
        else
            diff_expr = quote end
        end
        if !isempty(input.type_attr)
            for attr in op.attr
                if attr.name == input.type_attr
                    if isdefined(attr, :default_value)
                        convert_target = tf.Tensor{load_proto(attr.default_value)}
                        break
                    end
                end
            end
        end
        if input._type > 0 && haskey(proto_type_map, input._type)
            convert_target = tf.Tensor{proto_type_map[input._type]}
        end
        convert_expr = if isempty(input.number_attr) && isempty(input.type_list_attr)  # Scalar input
                :($sym=convert($(convert_target), $sym))
            else  # Array argument
                # :($sym=convert.($(convert_target), $sym))
                :($sym=[convert($(convert_target), x) for x in $sym])
            end
        push!(convert_block.args, quote
            $convert_expr
            $diff_expr
        end)
    end
    for type_set in values(type_sets)
        push!(convert_block.args, quote
            $(Expr(:tuple, type_set...)) = tf.tf_promote($(type_set...))
        end)
    end
    for (input_idx, input) in enumerate(op.input_arg)
        push!(input_block.args, quote
            tf.add_input(desc, $(inputs[input_idx]))
        end)
    end
    kwargs = Expr(:parameters)
    push!(kwargs.args, Expr(:kw, :name, nothing))
    attr_block = quote end
    for attr in op.attr
        is_internal_arg(attr) && continue
        name = Symbol(keyword_escape(attr.name))
        push!(kwargs.args, Expr(:kw, name, nothing))

        # Deal with attribute types like "list(int)"
        m = match(r"list(\(.*\))|(.*)", attr._type)
        t = m[1] !== nothing ? m[1] : m[2]

        t_map = Dict("int"=>:(Base.Int),
                     "bool"=>:(Base.Bool),
                     "tensor"=>:(TensorFlow.RawTensor),
                     "string"=>:(Base.String))
        t_target = get(t_map, t, :(Base.identity))
        if m[1] === nothing
            source = :($(t_target)($name))
        else
            source = :(map($t_target, $name))
        end
        if attr.name ∈ ["axis", "begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask", "component_index", "concat_dim"]
            push!(attr_block.args, quote
                if $name !== nothing
                    $name = $source - 1
                end
            end)
        elseif attr._type == "int" && attr.minimum == 0
            # info("Attribute $(op.name).$(attr.name) is likely an index and should be converted to 1-based indexing")
        end
        push!(attr_block.args, quote
            if $name !== nothing
                desc[$(attr.name)] = $source
            end
        end)
    end
    unshift!(inputs, kwargs)
    scalar_output = true
    if length(op.output_arg) > 1
        scalar_output = false
        n_output = length(op.output_arg)
    elseif length(op.output_arg) == 1
        output_arg = op.output_arg[1]
        if !isempty(output_arg.number_attr)
            scalar_output = false
            n_output = Symbol(output_arg.number_attr)
        end
    end
    output_block = if scalar_output
        :(tf.Tensor(tf.Operation(desc)))
    else
        quote
            out = tf.Tensor[]
            op = tf.Operation(desc)
            for out_idx in 1:$(n_output)
                push!(out, tf.Tensor(op, out_idx))
            end
            out
        end
    end
    expr = quote
        @tf.op function $(jl_name)($(inputs...))
            local desc
            tf.with_op_name(name, $(op.name)) do
                desc = tf.NodeDescription($(op.name))
                $convert_block
                $input_block
                $attr_block
            end
            $output_block
        end
    end
    posargs_str = join((arg.name for arg in op.input_arg), ", ")
    kwargs_str = []
    for arg in op.attr
        is_internal_arg(arg) && continue
        isdefined(arg, :default_value) || continue
        local default
        try
            default = load_proto(arg.default_value)
        catch err
            default = "?"
        end
        push!(kwargs_str, "$(arg.name)=$default")
    end
    if isempty(kwargs_str)
        kwargs_str = ""
    else
        kwargs_str = string("; ", join(kwargs_str, ", "))
    end

    sig = "$jl_name($(posargs_str)$(kwargs_str))"
    doc_str = string("     ", sig, "\n\n", op.summary, "\n\n", op.description)
    expr = unblock(MacroTools.flatten(MacroTools.striplines(expr)))
    OpFunc(expr, doc_str, jl_name)
end

function stringify_func(opfunc::OpFunc)
    s = string(opfunc.expr)
    docstring = replace(opfunc.docstring, "\$", "")
    doc_line = "\"\"\"\n$(docstring)\n\"\"\""
    lines = []
    "$doc_line\n$s"
end

stringify_func(op::tensorflow.OpDef) = stringify_func(to_function(op))

"""
    import_ops()

Autogenerates Julia functions for all TensorFlow operations defined in the
TensorFlow shared library.
"""
function import_ops()
    open(joinpath(dirname(@__FILE__), "ops/imported_ops.jl"), "w") do ops_file
        date = Dates.now()
        write(ops_file, """
        # Autogenerated on $date

        module Ops
        import TensorFlow
        const tf = TensorFlow
        """)
        for (name, op) in get_all_op_list()
            try
                f = to_function(op)
                s = stringify_func(f)
                write(ops_file, s)
                print(ops_file, "\n\n")
            catch err
                err_msg = sprint(showerror, err)
                warn("Could not import operation $name: $err_msg")
            end
        end
        write(ops_file, """
        end
        """)
    end
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

if VERSION >= v"0.6.0-dev.2123"
    include("ops/v6_ops.jl")
end

import .Ops: read_file
Base.read(::Type{Tensor}, filename) = read_file(filename)
