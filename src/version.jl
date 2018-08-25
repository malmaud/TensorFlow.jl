import Pkg

"""
    tf_version(kind=:backend)

Return the version number of the tensorflow library.

If `kind` is `:backend`, return the version of the TensorFlow binary loaded
into the Julia process.

If `kind` is `:julia`, return the version of the Julia TensorFlow package.

If `kind` is `:python`, return the version of the Python API.
"""
function tf_version(; kind=:backend)
    if kind == :backend
        res = @tfcall(:TF_Version, Cstring, ()) |> unsafe_string
    elseif kind == :python
        res = fetch(@py_proc py_tf[][:VERSION])
    elseif kind == :julia
        return Pkg.installed("TensorFlow")
    else
        @error("Kind '$kind' not recognized")
    end
    # Deal with version strings like "0.12.head"
    res = replace(res, r"\.head$"=>"")
    VersionNumber(res)
end

function version_check(v)
    if tf_version() < v
        @error("You have TensorFlow binary version $(tf_version()), but need version $v to use this functionality. Please upgrade with `Pkg.build(\"TensorFlow\").")
    end
end

"""
    py_version_check(;print_warning=true)

Returns true if the Python TensorFlow version is compatible with the TensorFlow binary used by the
Julia process.

If `print_warning` is true, print a warning with upgrade instructions if the versions are
incompatible.
"""
function py_version_check(;print_warning=true, force_warning=false)
    py_version = tf_version(kind=:python)
    lib_version = tf_version(kind=:backend)
    if (py_version < lib_version) || force_warning
        base_msg = "Your Python TensorFlow client version ($py_version) is below the TensorFlow backend version ($lib_version). This can cause various errors. Please upgrade your Python TensorFlow installation and then restart Julia."
        if PyCall.conda
            upgrade_msg = "You can upgrade by calling `using Conda; Conda.update();` from Julia."
        else
            upgrade_msg = "Typically, executing `pip install --upgrade tensorflow` from the command line will upgrade Python TensorFlow. You may need administrator privileges."
        end
        full_msg = "$(base_msg)\n$(upgrade_msg)"
        print_warning && warn(full_msg)
        return false
    end
    return true
end

macro tryshow(ex)
    quote
        try
            @show $(esc(ex))
        catch err
            println("Trying to evaluate ",
                    $(Meta.quot(ex)),
                    " but got error: ",
                    err
                    )
        end
        nothing
    end
end

function tf_versioninfo()
    println("Wording: Please copy-paste the entirely of the below output into any bug reports.")
    println("Note that this may display some errors, depending upon on your configuration. This is fine.")
    
    println()
    println("----------------")
    println("Library Versions")
    println("----------------")
    @tryshow ENV["TF_USE_GPU"]
    @tryshow ENV["LIBTENSORFLOW"]
    println()
    @tryshow tf_version(kind=:backend) 
    @tryshow tf_version(kind=:python)
    @tryshow tf_version(kind=:julia)  
    
    
    println()
    println("-------------")
    println("Python Status")
    println("-------------")
    @tryshow PyCall.conda
	@tryshow ENV["PYTHON"]
    @tryshow PyCall.PYTHONHOME
    @tryshow readstring(`pip --version`)
    @tryshow readstring(`pip3 --version`)
    
    println()
    println("------------")
    println("Julia Status")
    println("------------")
    versioninfo()
end
