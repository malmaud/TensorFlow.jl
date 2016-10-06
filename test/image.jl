sess = TensorFlow.Session()

blank_image = TensorFlow.constant(ones((5,5,5)))
result = run(sess, TensorFlow.image.flip_up_down(blank_image))
@test ones(5,5,5) == result

result = run(sess, TensorFlow.image.flip_left_right(blank_image))
@test ones(5,5,5) == result
