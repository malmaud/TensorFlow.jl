# This code is included into `ops/transformations.jl`
# These are higher-abstraction transformations about indexing and related functionality


"""
    Base.size(n::AbstractTensor; name="")

Returns the shape of the Tensor.
WARNING: this does not match the python TensorFlow `size` -- for that functionality, use `Base.length`
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
@define_unary Base.size shape
@op Base.size(n::AbstractTensor, i; name=nothing) = shape(n; name=name)[i]

"""
    Base.length(n::AbstractTensor; name="")

Returns the total number of elements in a Tensor.
(Like julia `Base.length` does for an `Array`)
This matchs python TensorFlow `size` operation
https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#size
"""
@define_unary Base.length Ops.size
@define_unary Base.lastindex length

Base.lastindex(x::AbstractTensor, dim) = size(x, dim)

struct TensorRange
    start::Tensor{Int32}
    stop::Tensor{Int32}
end
Base.first(tr::TensorRange)=tr.start
Base.last(tr::TensorRange)=tr.stop

Base.:(:)(start::AbstractTensor, stop::AbstractTensor) =  TensorRange(start, stop)
Base.:(:)(start, stop::AbstractTensor) =  TensorRange(start, stop)
Base.:(:)(start::AbstractTensor, stop) =  TensorRange(start, stop)


const Slice = Union{TensorRange, UnitRange, Colon}
const Index = Union{<:Integer, AbstractArray{<:Integer}, AbstractTensor{<:Integer}}


#For x[[true,false,true]] etc
function Base.getindex(params::AbstractTensor, indices::Union{Tensor{Bool}, AbstractArray{Bool}})
    boolean_mask(params, indices)
end

#For x[[1,2,3]] and x[2] etc
function Base.getindex(params::AbstractTensor, indices::Index)
    gather(params, indices)
end


#All other uses
function Base.getindex(params::AbstractTensor, ind1::Union{Slice, Index},  inds::Vararg{Union{Slice, Index}})
    indicies = [ind1, inds...]

    begins = Tensor{Int32}[]
    sizes = Tensor{Int32}[]
    singleton_dims = Int[]

    ### Begin  Subfunctions
    function proc_singleton!(ind)
        # Better be 0D
        push!(begins, ind)
        push!(sizes, 1)
        push!(singleton_dims, length(begins))
    end

    function proc_ind!(ind::Colon)
        push!(begins, 1)
        push!(sizes, -1) # Slice mark for go to end
    end

    function proc_ind!(ind::Union{UnitRange, TensorRange})
        #NOTE: end has (during code lowering) been replace with `lastindex(X)` or `size(X,d)` giving the actual size
        begin_ =  first(ind)
        push!(begins, begin_)
        end_ = last(ind)
        push!(sizes, end_ - begin_ + 1) # +1 because a slice 3:3 has length 1, and 4:5 has length 3 etc
    end

    function proc_ind!(ind::AbstractTensor)
        ind_shape = get_shape(ind)
        if ind_shape.rank_unknown
            #warn("Unknown rank tensor ($ind) used for indexing. Assuming 0D scalar.")
        elseif length(ind_shape.dims)!=0
            error("Non-OD scalar used for indexing ($ind). This form of mixed mode indexing is not currently supported.")
        end
        proc_singleton!(ind)
    end

    proc_ind!(ind::Integer) = proc_singleton!(ind)

    ### End Subfunctions

    for ind in indicies
        proc_ind!(ind)
    end

    with_op_name("PolyGetIndex") do
        begins_tensor = stack(begins)

        if length(singleton_dims) == length(begins) #Then we are not slicing any axes
            gather_nd(params, begins_tensor)
        else # We are slicing, at some axies
            sizes_tensor = stack(sizes)
            sliced = slice(params, begins_tensor, sizes_tensor)
            if length(singleton_dims)>0 # we are not slicing on all axies
                dropdims(sliced, dims=singleton_dims) # Drop singleton indexes
            else # we are slicing on all axies
                sliced
            end
        end
    end
end


