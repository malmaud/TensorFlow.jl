#Pkg.add("TensorFlow")
Pkg.clone("https://github.com/malmaud/TensorFlow.jl.git")
Pkg.build("TensorFlow")
Pkg.add("IJulia")
using TensorFlow
using IJulia
