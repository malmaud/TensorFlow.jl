const collections = Dict{Symbol, Any}()

function initialize_collections()
    collections[:Variables] = []
end

initialize_collections()

function add_to_collection(name, node)
    push!(collections[name], node)
end

function get_collection(name)
    return collections[name]
end
