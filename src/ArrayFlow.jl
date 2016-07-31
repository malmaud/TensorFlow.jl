module ArrayFlow

import Base: +, -, *, /, setindex!, run

include("constants.jl")
include("tensorflow.jl")
include("core.jl")
include("ops.jl")

end
