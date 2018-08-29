# The interface for "runnable" types

"""
`get_tensors(x)`

Return a vector of tensors that should be computed when `x` is passed to `run(::Session, ...)`.
`x` might be arbitrary nested in the array passed to `run`.
"""
function get_tensors end

"""
`build_output(tensors, values, position)`


"""
function build_output end

"""
`get_inputs(value, input_tensors, input_set)`

"""
function get_inputs end


function get_tensors(tensors::Union{Vector, Tuple})
    out = []
    for subtensor in tensors
        append!(out, get_tensors(subtensor))
    end
    out
end

function build_output(tensors::Union{Vector, Tuple}, values, pos=Ref(1))
    [build_output(subtensor, values, pos) for subtensor in tensors]
end


function get_tensors(tensor::Union{Tensor, Number})
    return [tensor]
end

get_tensors(tensor::IndexedSlices) = [tensor.values, tensor.indices]

function build_output(tensor, values, pos=Ref(1))
    out = values[pos[]]
    pos[] += 1
    out
end

function build_output(tensor::IndexedSlices, values, pos)
    out = IndexedSlicesValue(values[pos[]], values[pos[]+1])
    pos[] += 2
    out
end

function get_inputs(values, input_tensors, input_set=[])
    push!(input_set, values)
    input_set
end

function get_inputs(values, input_tensors::Vector)
    inputs = []
    for (value, subtensors) in zip(values, input_tensors)
        get_inputs(value, subtensors, inputs)
    end
    inputs
end

function build_input(tensor_map::Dict)
    input_tensors = Tensor[]
    input_values = []
    for (k,v) in tensor_map
        append!(input_tensors, get_tensors(k))
        append!(input_values, get_inputs(v,k))
    end

    input_tensors, input_values
end

struct ClosedSessionError <: Exception
end

function Base.show(io::IO, err::ClosedSessionError)
    print(io, "An operation was attempted on a closed TensorFlow session.")
end

function run(sess::Session, inputs, input_values, outputs, targets)
    #Low level run, without size checking, and type conversion etc.
    if sess.ptr == C_NULL
        throw(ClosedSessionError())
    end
    status = Status()
    output_values = fill(C_NULL, length(outputs))
    input_tensors = [RawTensor(x) for x in input_values]
    @tfcall(:TF_SessionRun, Cvoid,
    (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Cint, Ptr{Cvoid}, Cint, Ptr{Cvoid}, Ptr{Cvoid}),
        sess.ptr,
        C_NULL,
        inputs,
        [x.ptr for x in input_tensors],
        length(input_tensors),
        outputs,
        output_values,
        length(output_values),
        targets,
        length(targets),
        C_NULL,
        status.ptr)
    check_status(status)

    map(output_values) do x
        tensor = RawTensor(x)
        if ndims(tensor) == 0
            if eltype(tensor) == String
                convert(String, tensor)
            else
                convert(Number, tensor)
            end
        else
            convert(Array, tensor)
        end
    end
end

function cast_type(T, val::AbstractArray{<:Number})
    convert(Array{T}, val)
end

function cast_type(T, val::Number)
    convert(T, val)
end

cast_type(T, val) = val

tf_shape(x::Union{Number,String}) = TensorShape(Int[])
tf_shape(x::AbstractArray)        = TensorShape(collect(size(x)))

function run(sess::Session, outputs::AbstractVector, input_dict)
    output_map = Dict{Tensor, Tuple{Symbol, Int}}()
    output_ports = Port[]
    target_ptrs= Ptr{Cvoid}[]
    for tensor in get_tensors(outputs)
        if !haskey(output_map, tensor)
            if num_outputs(get_op(tensor)) == 0
                push!(target_ptrs, get_op(tensor).ptr)
                output_map[tensor] = (:target, length(target_ptrs))
            else
                push!(output_ports, Port(tensor))
                output_map[tensor] = (:output, length(output_ports))
            end
        end
    end
    input_ports = Port[]
    input_tensors, uncast_input_values = build_input(input_dict)
    input_values = []
    for (input_tensor, input_value) in zip(input_tensors, uncast_input_values)
        push!(input_values, cast_type(eltype(input_tensor), input_value))

    end
    input_ports = [Port(tensor) for tensor in input_tensors]
    unique_output_values = run(sess, input_ports, input_values, output_ports, target_ptrs)
    output_values = []
    for tensor in get_tensors(outputs)
        location = output_map[tensor]
        if location[1] == :target
            push!(output_values, nothing)
        elseif location[1] == :output
            push!(output_values, unique_output_values[location[2]])
        end
    end
    build_output(outputs, output_values)
end


"""
    run(sess::Session, output, input_dict::Dict)


Compute the result of one of more operations in the computation graph.
"""
function run(sess::Session, output, input_dict)
    res = run(sess, [output], input_dict)
    if length(res)==1
        return res[1]
    else
        return res
    end
end

run(sess::Session, outputs) = run(sess, outputs, Dict())

# Add ability to 'run' a numeric literal (for testing, generally)
run(sess::Session, output::Number) = run(sess, Tensor(output))
run(sess::Session, output::Array{<:Number}) = run(sess, Tensor(output))
