prefix = joinpath(ENV["HOME"], "tmp", "out")
gs_bucket = "malmaud-stuff"
version = "1.1.0"

run(`docker run -v /var/run/docker.sock:/var/run/docker.sock -e "prefix=$prefix" buildtf`)

full_names = []

function upload(dir, name)
    full_name = "tensorflow_$(name)_$(version).zip"
    push!(full_names, full_name)
    cd(joinpath(prefix, dir))
    run(`zip $full_name libtensorflow.so`)
    run(`gsutil cp $full_name gs://$gs_bucket`)
end

start_dir = pwd()

upload("cpu", "linux_cpu")
upload("gpu", "linux")
upload("mac", "mac")

cd(start_dir)

for name in full_names
    run(`gsutil acl ch -u AllUsers:R gs://$gs_bucket/$name`)
end
