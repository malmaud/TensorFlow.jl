# This code is included into `ops/transformations.jl`
# These are higher-abstraction transformations about indexing and related functionality


"""
Base.size(n::AbstractTensor; name="")
Returns the shape of the Tensor.
WARNING: this does not match the python TensorFlow `size` -- for that functionality, use `Base.length`
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
@op Base.size(n::AbstractTensor; name=nothing) = shape(n; name=name)
@op Base.size(n::AbstractTensor, i; name=nothing) = shape(n; name=name)[i]
# size(X, dim) must be implemented for indexing with X[..,end,...] to work

"""
Base.length(n::AbstractTensor; name="")
Returns the total number of elements in a Tensor.
(Like julia `Base.length` does for an `Array`)
This matchs python TensorFlow `size` operation
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
@op function Base.length(n::AbstractTensor; name=nothing)
    local desc
    with_op_name(name, "Size") do
        desc = NodeDescription("Size")
        add_input(desc, Tensor(n))
    end
    Tensor(Operation(desc), 1)
end
@op Base.endof(n::AbstractTensor; name=nothing) = length(n; name=name)
# endof(X) must be implemented for indexing with X[end] to work


immutable TensorRange
    start::Tensor{Int32}
    stop::Tensor{Int32}
end
Base.first(tr::TensorRange)=tr.start
Base.last(tr::TensorRange)=tr.stop

Base.colon(x,y::Tensor) = colon(Tensor(x), y)
Base.colon(x::Tensor, y) = colon(x, Tensor(y))
Base.colon(x::Tensor,y::Tensor) = TensorRange(x,y)

#For x[[1,2,3]] etc
function Base.getindex(params::AbstractTensor, indices)
    if eltype(indices) == Bool
        boolean_mask(params, indices)
    else
        gather(params, indices)
    end
end

#for slices eg X[1:end] etc
function Base.getindex(params::AbstractTensor, indices::Vararg{Union{TensorRange, UnitRange, Colon}})
    # This function is all about slices

    # TODO: Assign a name prefix to all the tensors made as art of this section, including constants
    begins = Tensor{Int32}[]
    sizes = Tensor{Int32}[]

    function proc_ind!(ind::Colon)
        push!(begins, 1)
        push!(sizes, -1) # Slice mark for go to end
    end
    function proc_ind!(ind::Union{UnitRange, TensorRange})
        #NOTE: end has now been replace with `endof(X)` or `size(X,d)` giving the actual size
        begin_ =  first(ind)
        push!(begins, begin_)
        end_ = last(ind)
        push!(sizes, end_ - begin_ + 1) # +1 because a slice 3:3 has length 1, and 4:5 has length 3 etc 
    end

    for ind in indices
        proc_ind!(ind)
    end

    begins_tensor = stack(begins)
    sizes_tensor = stack(sizes)
    slice(params, begins_tensor, sizes_tensor)
end


#For x[1,2,3] etc
function Base.getindex(params::AbstractTensor, indices...)
    inds::Vector = collect(indices) # Want Vector, not tuple. Could be a vector of Tensors though
    if eltype.(inds) ⊆ (Int32, Int64)
        gather_nd(params, inds)
    else
        error("julia style indexing is not currently supported for indicies $indices")
    end
end
