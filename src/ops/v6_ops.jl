# Operations code that uses syntax from Julia v0.6 and up.
# Mostly necessary to avoid type ambiguities with 'where' clauses.

function Base.fill(n::AbstractTensor, dims::Tuple{Vararg{Int64, N}} where N; kwargs...)
    invoke(fill, Tuple{AbstractTensor, Any}, n, dims; kwargs...)
end
