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


# Mixed Mode Indexing.
function getindex_polyfunction(params::AbstractTensor, indices...)
    begins = Tensor{Int32}[]
    sizes = Tensor{Int32}[]
    singleton_dims = Int[]

    function proc_singleton!(ind) #JULIA0.5BUG??. This function needs to be declared here (not mixed with the proc_ind!) or is just doesn't actually get declared. I haven't MWEd it yet
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
        #NOTE: end has now been replace with `endof(X)` or `size(X,d)` giving the actual size
        begin_ =  first(ind)
        push!(begins, begin_)
        end_ = last(ind)
        push!(sizes, end_ - begin_ + 1) # +1 because a slice 3:3 has length 1, and 4:5 has length 3 etc
    end

    function proc_ind!(ind::Tensor)
        ind_shape = get_shape(ind)
        if ind_shape.rank_unknown
            #warn("Unknown rank tensor ($ind) used for indexing. Assuming 0D scalar.")
        elseif length(ind_shape.dims)!=0
            error("Non-OD scalar used for indexing ($ind). This form of mixed mode indexing is not currently supported.")
        end
        proc_singleton!(ind)
    end

    proc_ind!(ind::Integer) = proc_singleton!(ind)

    for ind in indices
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
                squeeze(sliced, singleton_dims) # Squeeze singleton indexes
            else # we are slicing on all axies
                sliced
            end
        end
    end
end


# Union{Slice, Index} is the types for which getindex_polyfunction was designed
# Note: need to exclude Bools, because that is in Integer in 0.5
# This can be a lot cleaner all round in 0.6
const Slice = Union{TensorRange, UnitRange, Colon}
const Index = Union{Int16, Int32, Int64,
                          AbstractArray{Int16}, AbstractArray{Int32}, AbstractArray{Int64},
                          Tensor{Int16}, Tensor{Int32}, Tensor{Int64}}




#For x[[true,false,true]] etc
function Base.getindex(params::AbstractTensor, indices::Union{Tensor{Bool}, AbstractArray{Bool}})
    boolean_mask(params, indices)
end

#For x[[1,2,3]] and x[2] etc
function Base.getindex(params::AbstractTensor, indices::Index)
    gather(params, indices)
end


# If have one argument, then only use polyfunction if it is a Slice
function Base.getindex(params::AbstractTensor, ind::Slice)
    getindex_polyfunction(params, ind)
end

# If have 2+ argument can use the Polyfunction
function Base.getindex(params::AbstractTensor, ind1::Union{Slice, Index}, ind2::Union{Slice, Index},  inds::Vararg{Union{Slice, Index}})
    getindex_polyfunction(params, ind1, ind2, inds...)
end

function Base.getindex(params::AbstractTensor, inds::Vararg{AbstractTensor})
    getindex(params, map(Tensor, inds)...)
end
