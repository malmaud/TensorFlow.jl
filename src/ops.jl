using MacroTools
import Base: log, exp, +, -, *, /, ^, sin, cos, tan, asin, acos, atan, div, tanh, sqrt, abs, floor, ceil, floor, sign
import TensorFlow
const tf = TensorFlow # so know where op_funcs is defined

using MacroTools

if VERSION < v"0.6.0-"
    import Base: .*, .+, ./, .-, .^, .==
end



function tf_promote(t, x::Number)
    return Tensor(eltype(t)(x))
end

function tf_promote{T}(t, ::Type{Val{T}})  # Work around a^b->Val lowering
    return tf_promote(t, T)
end

tf_promote(t, x) = Tensor(x)

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
Reads and outputs the entire contents of the input filename.

Args:
  * filename: A `Tensor` of type `string`.
  * name: A name for the operation (optional).

Returns:
  A `Tensor` of type `string`.
"""
@op function read_file(filename; name=nothing)
    local desc
    with_op_name(name, "ReadFile") do
        desc = NodeDescription("ReadFile")
        add_input(desc, Tensor(filename))
    end
    Tensor(Operation(desc), 1)
end

Base.read(::Type{Tensor}, filename) = read_file(filename)

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

"""
    opname_to_jlname(name)

Converts a TensorFlow operation name `name` into a Julia
function name. Eg UnsortedSegmentSum->unsorted_segment_sum.
"""
function opname_to_jlname(name)
    keywords = ["const"]
    tokens = []
    pos = 1
    while true
        m = match(r"([A-Z]{2,})|(V\d+$)|(\d+$)|(\d+.)|([A-Z][a-z]*)", name, pos)
        m === nothing && break
        lowered = lowercase(m.match)
        if lowered in keywords
            lowered = "$(lowered)_"
        end
        push!(tokens, lowered)
        pos = m.offset + length(m.match)
    end
    Symbol(join(tokens, "_"))
end

function to_function(op::tensorflow.OpDef)
    jl_name = Symbol(lowercase(op.name))
    jl_name = opname_to_jlname(op.name)
    inputs = []
    input_block = quote end
    for input in op.input_arg
        sym = gensym()
        push!(inputs, sym)
        if input.type_attr == "Index"
            convert_target = tf.Tensor{Int32}
            diff_expr = quote
                converted = converted - 1
            end
        else
            convert_target = tf.Tensor{Any}
            diff_expr = quote end
        end
        push!(input_block.args, quote
            if isa($sym, AbstractArray)
                converted = convert.($(convert_target), $sym)
            else
                converted = convert($(convert_target), $sym)
            end
            $diff_expr
            tf.add_input(desc, converted)
        end)
    end
    kwargs = Expr(:parameters)
    push!(kwargs.args, Expr(:kw, :name, nothing))
    attr_block = quote end
    for attr in op.attr
        isdefined(attr, :default_value) || continue
        push!(kwargs.args, Expr(:kw, Symbol(attr.name), nothing))
        push!(attr_block.args, quote
            if $(Symbol(attr.name)) !== nothing
                desc[$(attr.name)] = $(Symbol(attr.name))
            end
        end)
    end
    unshift!(inputs, kwargs)
    expr = quote
        function $(jl_name)($(inputs...))
            local desc
            tf.with_op_name(name, $(op.name)) do
                desc = tf.NodeDescription($(op.name))
                $input_block
                $attr_block
            end
            tf.Tensor(tf.Operation(desc))
        end
    end
    posargs_str = join((arg.name for arg in op.input_arg), ", ")
    kwargs_str = []
    for arg in op.attr
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
    OpFunc(expr, doc_str, jl_name)
end

function stringify_func(opfunc::OpFunc)
    s = sprint(show, opfunc.expr)
    noquote = split(s, "\n")[2:end-1]
    docstring = replace(opfunc.docstring, "\$", "")
    lines = ["\"\"\"\n$(docstring)\n\"\"\""]
    for line in noquote
        line = replace(line, r"##", "v")
        line = replace(line, r"#.*$", "")
        push!(lines, line[5:end])
    end
    join(lines, "\n")
end

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
