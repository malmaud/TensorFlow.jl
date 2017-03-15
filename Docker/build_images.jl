user = ENV["USER"]
run(`docker build --no-cache -t $user/julia:tf cpu`)
run(`docker build --no-cache -t $user/julia:tf_gpu gpu`)
run(`docker push $user/julia:tf`)
run(`docker push $user/julia:tf_gpu`)
