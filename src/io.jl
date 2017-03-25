module io

export
WholeFileReader,
TextLineReader

import TensorFlow
import TensorFlow: @op
const tf = TensorFlow

include("io/readers.jl")

end
