function get_tensors(tensors::Vector)
    out = []
    for subtensor in tensors
        append!(out, get_tensors(subtensor))
    end
    out
end

function build_output(tensors::Vector, values, pos=Ref(1))
    [build_output(subtensor, values, pos) for subtensor in tensors]
end

function get_tensors(tensor::Union{Tensor, Number})
    return [tensor]
end

function build_output(tensor::Tensor, values, pos)
    out = values[pos[]]
    pos[] += 1
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

function run(sess::Session, inputs, input_values, outputs, targets)
    status = Status()
    output_values = fill(C_NULL, length(outputs))
    input_tensors = [RawTensor(_) for _ in input_values]
    ccall((:TF_SessionRun, LIBTF), Void,
    (Ptr{Void}, Ptr{Void}, Ptr{Void}, Ptr{Void}, Cint, Ptr{Void}, Ptr{Ptr{Void}}, Cint, Ptr{Void}, Cint, Ptr{Void}, Ptr{Void}),
        sess.ptr,
        C_NULL,
        inputs,
        [_.ptr for _ in input_tensors],
        length(input_tensors),
        outputs,
        output_values,
        length(output_values),
        targets,
        length(targets),
        C_NULL,
        status.ptr)
    check_status(status)
    as_native = tensor->begin
        if ndims(tensor) == 0
            if eltype(tensor) == String
                String(tensor)
            else
                Number(tensor)
            end
        else
            Array(tensor)
        end
    end
    return [as_native(RawTensor(_)) for _ in output_values]
end

function cast_type{Q<:Number}(T, val::Array{Q})
    convert(Array{T}, val)
end

function cast_type(T, val::Number)
    convert(T, val)
end

cast_type(T, val) = val

function run(sess::Session, outputs::AbstractVector, input_dict)
    output_map = Dict{Tensor, Tuple{Symbol, Int}}()
    output_ports = Port[]
    target_ptrs= Ptr{Void}[]
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
