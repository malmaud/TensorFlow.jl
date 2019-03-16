######
# Functions for importing operations defined in libtensorflow into
# Julia functions.
######

using MacroTools
using Dates

struct OpFunc
    expr::Expr
    eager_expr::Expr
    dispatch_expr::Expr
    docstring::String
    name::Symbol
end

"""
    keyword_escape(string::AbstractString)

If `string` is not allowed as a Julia variable identifier, suffix it with a `_`.
Otherwise, return it unchanged.
"""
function keyword_escape(s)
    keywords = ["const", "type", "while", "for", "if"]
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
            next_char = name[idx + 1]
            if idx < length(name) - 1
                next_next_char = name[idx + 2]
                if isuppercase(cur_char) && isuppercase(next_char) && islowercase(next_next_char)
                    word_end = true
                end
            end
            if islowercase(cur_char) && isuppercase(next_char)
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

"""
    is_internal_arg(argname)

Returns `true` if the given operation attribute is not meant to be supplied
by the user and `false` otherwise.
"""
function is_internal_arg(arg)
    arg._type == "type" && occursin(r"^T", arg.name)
end

function to_function(op::tensorflow.OpDef)
    jl_name = Symbol(lowercase(op.name))
    jl_name = opname_to_jlname(op.name)
    inputs = []
    input_block = quote end
    convert_block = quote end
    type_sets = Dict{String,Vector{Symbol}}()
    for (i, input) in enumerate(op.input_arg)
        sym = Symbol("$(input.name)_")
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
            convert_target = tf.Tensor{(proto_type_map[input._type])}
        end
        convert_expr = if isempty(input.number_attr) && isempty(input.type_list_attr)  # Scalar input
            :($sym = convert($(convert_target), $sym))
        else  # Array argument
                # :($sym=convert.($(convert_target), $sym))
            :($sym = [convert($(convert_target), x) for x in $sym])
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

        t_map = Dict("int" => :(Base.Int),
                     "bool" => :(Base.Bool),
                     "tensor" => :(TensorFlow.RawTensor),
                     "string" => :(Base.String))
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
    t_block = []
    for (i, input_arg) in enumerate(op.input_arg)
        if has_field(input_arg, :type_attr)
            type_attr = input_arg.type_attr
            if length(type_attr) > 0
                code = quote
                    desc[$type_attr]  = tf.data_type($(inputs[i]))
                end
                push!(t_block, code)
            end
        end
    end
    pushfirst!(inputs, kwargs)
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
    eager_output_block  = scalar_output ? :(return res[1]) : :(return res)
    graph_name = Symbol("$(jl_name)_graph")
    eager_name = Symbol("$(jl_name)_eager")
    expr = quote
        function $graph_name($(inputs...))
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

    eager_convert_block = []
    for input in inputs[2:end]
        c = :($input = convert(tf.EagerTensor, $input))
        push!(eager_convert_block, c)
    end

    eager_expr = quote
        function $eager_name($(inputs...))
            desc = tf.EagerOp($(op.name))
            # $convert_block
            $(eager_convert_block...)
            $input_block
            $attr_block
            $(t_block...)
            res = tf.execute(desc)
            node = tf.TapeNode($jl_name, [$(inputs[2:end]...)], $(inputs[1].args...), res)
            if length(res) >= 1
                tf.add_node(res[1], node)
                $eager_output_block
            end
        end
    end

    call_kw_params = Expr(:parameters)
    for arg in inputs[1].args
        push!(call_kw_params.args, Expr(:kw, arg.args[1], arg.args[1]))
    end
    call_args = [call_kw_params; inputs[2:end]]
    dispatch_expr = quote
        @tf.op function $jl_name($(inputs...))
            if tf.in_eager_mode()
                $(eager_name)($(call_args...))
            else
                $(graph_name)($(call_args...))
            end
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
        if default === nothing # Not sure why this is happening. It's happening for dropout
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
    doc_str = string("     ", sig, "\n\n",
                     escape_string(op.summary)) #TODO Workout how to get descriptions for docstrings
    OpFunc(expr, eager_expr, dispatch_expr, doc_str, jl_name)
end

"""
    stringify_func(opfunc::OpFunc)

Returns a text (unparsed) version of the given `OpFunc`, suitable to be
parsed by Julia's parser.

The function is returned with a triple-quoted docstring.
"""
function stringify_func(opfunc::OpFunc)
    expr = quote
        $(opfunc.expr)
        $(opfunc.eager_expr)
        $(opfunc.dispatch_expr)
    end
    # MacroTools.flatten seems to have a bug that's causins an invalid expression for 'NoOp'
    # expr = (MacroTools.flatten(MacroTools.striplines(expr)))
    expr = MacroTools.striplines(expr)

    s = string(expr)
    docstring = replace(opfunc.docstring, "\$" => "")
    doc_line = "\"\"\"\n$(docstring)\n\"\"\""
    "$doc_line\n$s\n"
end

stringify_func(op::tensorflow.OpDef) = stringify_func(to_function(op))


function load_default_imports()
    path = joinpath(@__DIR__, "../deps/default_imports.txt")
    names = readlines(path)
    filtered = [name for name in names if !occursin(r"^#", name)]
    return filtered
end

"""
    import_ops(op_names)

Autogenerates Julia functions for all TensorFlow operations defined in the
TensorFlow shared library.
"""
function import_ops(op_names)
    ops = Dict(get_all_op_list())
    open(joinpath(@__DIR__, "ops/imported_ops.jl"), "w") do ops_file
        date = Dates.now()
        write(ops_file, """
        # Autogenerated on $date

        module Ops
        import TensorFlow
        const tf = TensorFlow
        import TensorFlow: Tensor
        """)
        for name in op_names
            op = ops[name]
            # try
                f = to_function(op)
                s = stringify_func(f)
                write(ops_file, s)
                print(ops_file, "\n\n")
            # catch err
                # err_msg = sprint(showerror, err)
                # @warn("Could not import operation $name: $err_msg")
            # end
        end
        write(ops_file, """
        end
        """)
    end
end

"""
    import_ops()

Load the default set of TensorFlow operations
"""
import_ops() = import_ops(load_default_imports())

"""
    import_all_ops()

Import all defined operations
"""
function import_all_ops()
    ops = get_all_op_list()
    names = [op[1] for op in ops]
    import_ops(names)
end

"""
    import_op(op_name)

Import the given TensorFlow option into Julia. The name should correspond to
a key in the return value of `get_all_op_list()`, which gives the names of
TensorFlow operations defined in your version of the TensorFlow C binary.

`op_name` is the name of the TensorFlow operation, such as "Add".

Returns a reference to a Julia function corresponding to the operation.
"""
function import_op(name)
    jl_name = opname_to_jlname(name)
    mod = TensorFlow.Ops
    if jl_name ∉ names(mod, all = true)
        ops = Dict(get_all_op_list())
        op = ops[name]
        op_desc = to_function(op)
        Core.eval(Ops, op_desc.expr)
    else
        @warn("Import Skipped: tried to import op $name as $(mod).$(jl_name), but that already exists.")
    end

    return getfield(Ops, jl_name)
end
