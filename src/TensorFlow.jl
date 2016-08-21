module TensorFlow

include("constants.jl")
include("tensorflow_protos.jl")
include("core.jl")
include("variable.jl")
include("ops.jl")
include("train.jl")

function __init__()
    if myid() == 1
        set_def_graph(Graph())
        spawn_py_process()
    end
end

end
