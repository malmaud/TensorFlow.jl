using Test

sess = TensorFlow.Session(TensorFlow.Graph())

blank_image = TensorFlow.constant(ones((5,5,5)))
result = run(sess, TensorFlow.image.flip_up_down(blank_image))
@test ones(5,5,5) == result

result = run(sess, TensorFlow.image.flip_left_right(blank_image))
@test ones(5,5,5) == result

white_png = TensorFlow.constant(ones(UInt8,5,5,1))
result = run(sess, TensorFlow.image.decode_png(TensorFlow.image.encode_png(white_png)))
@test result == ones(5,5,1)
