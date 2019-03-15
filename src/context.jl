abstract type Context
end

struct ContextStack
    contexts::Vector{Context}
end

ContextStack() = ContextStack(Context[])

function Base.push!(stack::ContextStack, context::Context)
    push!(stack.contexts, context)
end

function Base.pop!(stack::ContextStack)
    pop!(stack.contexts)
end

function default_context()
    return [ExecutionMode(eager=false)]
end

function context_value(context_type)
    return global_context[context_type]
end

function Base.getindex(c::ContextStack, context_type)
    value = nothing
    for context in c.contexts
        if isa(context, context_type)
            value = context
        end
    end
    return value
end

function with_context(block, ctx)
    push!(global_context, ctx)
    res = block()
    # This assumes the block doesn't adjust the context. We should pop explicitly the pushed context.
    pop!(global_context)
    return res
end

const global_context = ContextStack()
