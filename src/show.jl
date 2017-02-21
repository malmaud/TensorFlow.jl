using Juno
import Juno: Tree, Row, fade, interleave

@render Juno.Inline t::Tensor begin
  s = get_shape(t)
  shape = s.rank_unknown ? [fade("unknown")] :
    interleave(map(dim -> get(dim, fade("?")), s.dims), fade("×"))
  Tree(Row(fade(try string(eltype(t)," ") catch e "" end),
           Juno.span(".constant.support.type", "Tensor "),
           shape...),
       [Text("name: $(node_name(t.op))"),
        Text("index: $(t.value_index)")])
end

function Base.show(io::IO, s::Status)
    msg = ccall((:TF_Message, LIBTF), Cstring, (Ptr{Void},), s.ptr) |> unsafe_string
    print(io, @sprintf("Status: %s", msg))
end

function Base.show(io::IO, err::TFException)
    println(io, @sprintf("Tensorflow error: %s", string(err.status)))
end

function Base.show(io::IO, s::Session)
    print(io, "Session($(pointer_from_objref(s)))")
end

function Base.show(io::IO, t::RawTensor)
    print(io, "RawTensor: ")
    if ndims(t) == 0
        if eltype(t) == String
            show(io, String(t))
        else
            show(io, Number(t))
        end
    else
        show(io, Array(t))
    end
end

function Base.show(io::IO, n::Operation)
    print(io, "<Operation '$(node_name(n))'>")
end

function Base.show(io::IO, t::Tensor)
    local dtype
    try
        dtype = eltype(t)
    catch
        dtype = "?"
    end
    s = get_shape(t)
    if s.rank_unknown
        shape = "unknown"
    else
        dims = String[]
        for dim in s.dims
            if isnull(dim)
                push!(dims, "?")
            else
                push!(dims, string(get(dim)))
            end
        end
        shape = string("(", join(dims, ", "), ")")
    end
    print(io, "<Tensor $(node_name(t.op)):$(t.value_index) shape=$(shape) dtype=$(dtype)>")
end

function Base.show(io::IO, desc::tensorflow.NodeDef)
    # TODO: complete this
    has_field(desc, :name) && println(io, "name: ", desc.name)
    has_field(desc, :op) && println(io, "op: ", desc.op)
    if has_field(desc, :input)
        for input_name in desc.input
            println(io, "input: ", input_name)
        end
    end
    if has_field(desc, :attr)
        for (attr_name, attr_value) in desc.attr
            println(io, "attr {")
            println(io, "  key: ", attr_name)
            println(io, "  value {")
            print(io, "    ")
            if has_field(attr_value, :_type)
                println(io, "type: $(proto_type_map[attr_value._type])")
            elseif has_field(attr_value, :s)
                println(io, "string: $(String(attr_value.s))")
            elseif has_field(attr_value, :i)
                println(io, "int: $(attr_value.i)")
            elseif has_field(attr_value, :b)
                println(io, "bool: $(attr_value.b)")
            elseif has_field(attr_value, :f)
                println(io, "float: $(attr_value.f)")
            elseif has_field(attr_value, :tensor)
                t = attr_value.tensor
                println(io, "dtype: $(proto_type_map[t.dtype])")
                sep = "    "
                print(io, sep, "shape: ")
                println(io, [_.size for _ in t.tensor_shape.dim])
                print(io, sep, "content: ")
                show_tensor = k->begin
                    f = getfield(t, k)
                    if length(f) > 0
                        println(io, f)
                        return true
                    end
                    return false
                end
                for v in [:float_val, :double_val, :int_val, :int64_val, :bool_val, :half_val, :string_val, :tensor_content]
                    if show_tensor(v)
                        break
                    end
                end
            elseif has_field(attr_value, :shape)
                print(io, "shape: ")
                println(io, [_.size for _ in attr_value.shape.dim])
            elseif has_field(attr_value, :list)
                list = attr_value.list
                if has_field(list, :s)
                    println(io, "string: $(String.(list.s))")
                end
            end
            println(io, "  }")
            println(io, "}")
        end
    end
end
