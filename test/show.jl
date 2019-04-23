using TensorFlow
using Test

@testset "Tensorboard" begin
    if Sys.iswindows()
        println("Use the python Tensorboard. Setting up Julia isn't worth the effort since you need a working python 3.x anyways.")       
    else
        TensorFlow.get_tensorboard()
    end
end
