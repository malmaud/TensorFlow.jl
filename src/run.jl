
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

function build_key_map{T}(elems::AbstractVector, output_map::Dict{T, Int})
    out = []
    for elem in elems
        push!(out, build_key_map(elem, output_map))
    end
    out
end

function build_key_map{T}(elem::T, output_map::Dict{T, Int})
    if haskey(output_map, elem)
        output_map[elem]
    else
        id = length(output_map)+1
        output_map[elem] = id
        id
    end
end

function build_key_map(elems)
    d = Dict{Tensor, Int}()
    out = build_key_map(elems, d)
    out, d
end

function use_key_map(shapes::AbstractVector, value_map)
    out = []
    for shape in shapes
        push!(out, use_key_map(shape, value_map))
    end
    out
end

function use_key_map(value, value_map)
    get(value_map, value, [])
end

function build_input_map(d_in::Dict, d_out)
    for (key, value) in d_in
        build_input_map(key, value, d_out)
    end
    d_out
end

function build_input_map(key::AbstractVector, value, d_out)
    for (key_elem, value_elem) in zip(key, value)
        build_input_map(key_elem, value_elem, d_out)
    end
    d_out
end

function build_input_map(key, value, d_out)
    d_out[key] = value
    d_out
end

build_input_map(d_in) = build_input_map(d_in, Dict())

function run(sess::Session, outputs::AbstractVector, input_dict)
    isempty(outputs) && return []
    inputs = Port[]
    input_values = []
    input_dict = build_input_map(input_dict)
    for (input, value) in input_dict
        push!(inputs, Port(input))
        push!(input_values, map(eltype(input), value))
    end
    real_outputs = Tensor[]
    targets = Ptr{Void}[]
    id_map = Dict{Int, Any}()
    output_shape, output_map = build_key_map(outputs)
    for (tensor, id) in output_map
        if num_outputs(get_op(tensor)) == 0
            push!(targets, get_op(tensor).ptr)
        else
            push!(real_outputs, tensor)
            id_map[length(real_outputs)] = id
        end
    end
    output_ports = map(Port, real_outputs)
    result = run(sess, inputs, input_values, output_ports, targets)
    result_map = Dict{Int, Any}()
    for (id, res) in enumerate(result)
        result_map[id_map[id]] = res
    end
    use_key_map(output_shape, result_map)
end

"""
Compute the result of one of more operations in the computation graph.
"""
function run(sess::Session, output::Tensor, input_dict)
    res = run(sess, [output], input_dict)
    if length(res)==1
        return res[1]
    else
        return res
    end
end

run(sess::Session, outputs) = run(sess, outputs, Dict())
