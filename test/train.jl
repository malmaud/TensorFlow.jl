using TensorFlow
using Base.Test

@testset "train.save and train.resore" begin
	let
	session = Session(Graph())
		x = get_variable("x", [], Float32)
		run(session, assign(x, 5.f0))
		saver = train.Saver()
		train.save(saver, session, "weights")
	end
	
	let
		session = Session(Graph())
		@tf x = get_variable([], Float32)
		saver = train.Saver()
		train.restore(saver, session, "weights")
		@test run(session, x) == 5.0f0
	end
end
