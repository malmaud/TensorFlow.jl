module train

import ..TensorFlow: Node, get_def_graph, gradients, assign

abstract Optimizer

type GradientDescentOptimizer <: Optimizer
    learning_rate::Node
    name::String
end

GradientDescentOptimizer(learning_rate; name="") = GradientDescentOptimizer(Node(learning_rate), name)

function compute_gradients(optimizer::GradientDescentOptimizer, loss, var_list=nothing)
    if var_list === nothing
        var_list = get_def_graph().collections[:Variables]
    end
    zip(gradients(loss, var_list), var_list) |> collect
end

function apply_gradients(optimizer::GradientDescentOptimizer, grads_and_vars; global_step=nothing, name="")
    ops = Node[]
    for (grad, var) in grads_and_vars
        push!(ops, assign(var, var - optimizer.learning_rate.*grad))
    end
    return ops
end

function minimize(optimizer::Optimizer, loss; global_step=nothing, var_list=nothing, name="")
    grads = compute_gradients(optimizer, loss, var_list)
    apply = apply_gradients(optimizer, grads; global_step=global_step, name=name)
end

end
