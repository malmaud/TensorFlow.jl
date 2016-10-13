module io

export
WholeFileReader,
TextLineReader

import TensorFlow
const tf = TensorFlow

include("io/readers.jl")

end
