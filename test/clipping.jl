using TensorFlow
using Base.Test

sess = TensorFlow.Session(TensorFlow.Graph())
a_raw = rand(10, 10)
a = TensorFlow.constant(a_raw)
result = run(sess, clamp(a, 0.3, 0.7))
@test clamp.(a_raw, 0.3, 0.7) == result

a_raw = rand(10)
a = TensorFlow.constant(a_raw)
result = run(sess, TensorFlow.clip_by_norm(a, 1.))
@test normalize(a_raw) ≈ result

b_raw = rand(10)
b = TensorFlow.constant(b_raw)
result = run(sess, TensorFlow.global_norm([a,b]))
gn = hypot(norm(a_raw), norm(b_raw))
@test gn ≈ result
