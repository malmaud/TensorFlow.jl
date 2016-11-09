import Base: log, exp, +, -, *, /, .*, .+, ./, .-, ^, .^, sin, cos, tan, asin, acos, atan, div, tanh, sqrt, abs, floor, .==, ceil, floor, sign

function tf_promote(t, x::Number)
    return Tensor(eltype(t)(x))
end

tf_promote(t, x) = Tensor(x)


convert_number(t, n) = n
convert_number(t, x::Number) =  t(x)

to_tensor(x::Union{Number, String, AbstractTensor}) = Tensor(x)
to_tensor(x::AbstractArray) = Tensor(x)

to_list(x::AbstractArray) = x
to_list(x) = [x]

macro not_implemented(f)
    if f.head != :function
        error("Invalid use of not_implemented")
    end
    func_name = f.args[1].args[1]
    quote
        function $(esc(func_name))(args...; kwargs...)
            error("Not implemented yet")
        end
    end
end

const name_idx = Dict{String,Int}()

function capitalize(s)
    string(uppercase(s[1]), s[2:end])
end

capitalize(s::Symbol) = capitalize(string(s))

function get_name(name="node")
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
function placeholder(dtype; name="placeholder", shape=nothing)
    local node
    with_op_name(name) do
        graph = get_def_graph()
        desc = NodeDescription("Placeholder")
        desc["dtype"] = dtype
        node = Operation(desc)
        if shape===nothing
            graph.shapes[name] = ShapeInference.TensorShape(nothing)
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
function read_file(filename; name="ReadFile")
    local desc
    with_op_name(name) do
        desc = NodeDescription("ReadFile")
        add_input(desc, Tensor(filename))
    end
    Tensor(Operation(desc), 1)
end

Base.read(::Type{Tensor}, filename) = read_file(filename)


include("ops/math.jl")
include("ops/sequences.jl")
include("ops/control_flow.jl")
include("ops/logical.jl")
include("ops/debugging.jl")
include("ops/comparison.jl")
include("ops/transformations.jl")
include("ops/nn.jl")
include("ops/image.jl")
include("ops/summaries.jl")
include("ops/queues.jl")
include("ops/clipping.jl")
