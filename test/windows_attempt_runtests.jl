# This is a temporary testing file, for purposes of debugging what is going on with windows

julia = Cmd([joinpath(JULIA_HOME, "julia.exe")])

tests = [
    "math.jl",
    "meta.jl",
    "control.jl",
    "nn.jl",
    "training.jl",
]

for test in tests
    println(test)

    try
        run(`$julia $test`)
    catch err
        warn(err)
    end

    println("")
    println("-------------")
    println("")
end
