module io

export
WholeFileReader,
TetLineReader

import TensorFlow
const tf = TensorFlow

include("io/readers.jl")

end
