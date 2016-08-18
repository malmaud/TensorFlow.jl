module train

import ..TensorFlow: Node, get_def_graph, gradients, assign, variable_scope, ConstantInitializer, node_name, get_variable, get_shape

abstract Optimizer

function minimize(optimizer::Optimizer, loss; global_step=nothing, var_list=nothing, name="")
    if name == ""
        name = optimizer.name
    end
    grads = compute_gradients(optimizer, loss, var_list)
    apply = apply_gradients(optimizer, grads; global_step=global_step, name=name)
end

function compute_gradients(optimizer::Optimizer, loss, var_list=nothing)
    if var_list === nothing
        var_list = get_def_graph().collections[:Variables]
    end
    zip(gradients(loss, var_list), var_list) |> collect
end

macro advance_step()
    quote
        if global_step !== nothing
            push!(ops, assign(global_step, global_step+1))
        end
    end
end

type GradientDescentOptimizer <: Optimizer
    learning_rate::Node
    name::String
end

GradientDescentOptimizer(learning_rate; name="descent") = GradientDescentOptimizer(Node(learning_rate), name)

function apply_gradients(optimizer::GradientDescentOptimizer, grads_and_vars; global_step=nothing, name="descent")
    ops = Node[]
    for (grad, var) in grads_and_vars
        push!(ops, assign(var, var - optimizer.learning_rate.*grad))
    end
    @advance_step
    return ops
end

type MomentumOptimizer <: Optimizer
    learning_rate::Node
    momentum::Node
    name::String
end

MomentumOptimizer(learning_rate, momentum; name="momentum") = MomentumOptimizer(learning_rate, momentum, name)

function apply_gradients(optimizer::MomentumOptimizer, grads_and_vars; global_step=nothing, name="momentum")
    ops = Node[]
    @advance_step
    for (grad, var) in grads_and_vars
        local momentum
        variable_scope(name) do
            variable_scope(node_name(var)) do
                momentum = get_variable("momentum", get_shape(var), eltype(var), initializer=ConstantInitializer(0.0))
            end
        end
        step = -optimizer.learning_rate .* grad + optimizer.momentum .* momentum
        push!(ops, assign(var, var + step))
        push!(ops, assign(momentum, step))
    end
    return ops
end


end
