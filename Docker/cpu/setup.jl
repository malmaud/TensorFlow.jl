Pkg.add("TensorFlow")
Pkg.add("IJulia")
Pkg.checkout("TensorFlow")
Pkg.build("TensorFlow")
using TensorFlow
using IJulia
