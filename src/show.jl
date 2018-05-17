using Juno
using PyCall
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
    msg = @tfcall(:TF_Message, Cstring, (Ptr{Void},), s.ptr) |> unsafe_string
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

function Base.show(io::IO, t::Tensor{T}) where T
    @assert T==eltype(t)

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
    print(io, "<Tensor $(node_name(t.op)):$(t.value_index) shape=$(shape) dtype=$(T)>")
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
                println(io, [x.size for x in t.tensor_shape.dim])
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
                println(io, [x.size for x in attr_value.shape.dim])
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


############################################################################################################
# TensorBoard
############################################################################################################


struct Tensorboard
    proc::Base.Process
    logdir::String
    port::Int
end

const tensorboard = Ref{Tensorboard}()

function Base.close(board::Tensorboard)
    kill(board.proc, 2)
end

"""
    find_tensorboard()

Return the path to the tensorboard executable.

Checks everywhere in the system path, as well in the binary python directory
associated with the Conda.jl installation, should it exist.
"""
function find_tensorboard()
    path = nothing
    dirs = split(ENV["PATH"], ":")
    if PyCall.conda
        push!(dirs, PyCall.Conda.BINDIR)
    end
    for dir in dirs
        loc = joinpath(dir, "tensorboard")
        if isfile(loc)
            path = loc
        end
    end
    return path
end

function get_tensorboard(logdir=nothing)
    if isdefined(tensorboard, :x)
        port = tensorboard[].port + 1
    else
        port = 6006
    end
    if logdir === nothing
        logdir = mktempdir()
    end
    path = find_tensorboard()
    if path === nothing
        error("The tensorboard binary was not found. Make sure `tensorboard` is in your system path.")
    end
    _, proc = open(`$path --logdir=$logdir --port=$port`)
    tensorboard[] = Tensorboard(proc, logdir, port)
    atexit() do
        close(tensorboard[])
    end
    sleep(3)
    tensorboard[]
end

function open_url(url)
    cmd = nothing
    if is_apple()
        cmd = `open $url`
    elseif is_unix()
        cmd = `xdg-open $url`
    end
    cmd === nothing || run(cmd)
end

"""
    visualize(x)

Visualize the given TensorFlow object `x`.

Generally this will work by loading TensorBoard and opening a web
browser.
"""
function visualize end

@with_def_graph function visualize(g::Graph)
    tensorboard = get_tensorboard()
    writer = summary.FileWriter(tensorboard.logdir, graph=g)
    visualize(writer)
    close(writer)
end

function visualize(writer::summary.FileWriter)
    tensorboard = get_tensorboard(writer.logdir)
    open_url("http://localhost:$(tensorboard.port)/#graphs")
end
