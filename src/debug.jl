@with_def_graph function show_op_names(g::Graph)
    for (i, node) in enumerate(get_def(g).node)
        println("$(i): $(node.name)")
        for (j, input) in enumerate(node.input)
            println("    $(j): $(input)")
        end
    end
end
