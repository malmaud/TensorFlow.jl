sess=Session()



v=Variable([1,2])
init = initialize_all_variables()
extend_graph(sess)

run(sess, init)
run(sess,v)
