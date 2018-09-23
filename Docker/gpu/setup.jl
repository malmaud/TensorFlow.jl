#Pkg.add("TensorFlow")
using Pkg
Pkg.clone(PackageSpec("https://github.com/malmaud/TensorFlow.jl.git", rev = "master"))
Pkg.build("TensorFlow")
Pkg.add("IJulia")
using TensorFlow
using IJulia
