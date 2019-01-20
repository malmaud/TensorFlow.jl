mutable struct EagerContext
    ptr::Ptr{Cvoid}

    function EagerContext()
        options = @tfcall(:TFE_NewContextOptions, Ptr{Cvoid}, ())
        status = Status()
        context = @tfcall(:TFE_NewContext, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), options, status)
        check_status(status)
        this = new(context)
        finalizer(this) do self
            @tfcall(:TFE_DeleteContext, Cvoid, (Ptr{Cvoid},), self.ptr)
        end
        @tfcall(:TFE_DeleteContextOptions, Cvoid, (Ptr{Cvoid},), options)
        return this
    end
end


mutable struct TensorHandle
    ptr::Ptr{Cvoid}

    function TensorHandle(tensor)
        status = Status()
        ptr = @tfcall(:TFE_NewTensorHandle, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), tensor.ptr, status)
        check_status(status)
        this = new(ptr)
        finalizer(this) do self
            @tfcall(:TFE_DeleteTensorHandle, Cvoid, (Ptr{Cvoid},), self.ptr)
        end
        return this
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::TensorHandle) = h.ptr


function device_name(h::TensorHandle)
    status = Status()
    c_name = @tfcall(:TFE_TensorHandleDeviceName, Cstring, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    return unsafe_string(c_name)
end

function resolve(h::TensorHandle)
    status = Status()
    ptr = @tfcall(:TFE_TensorHandleResolve, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    tensor = RawTensor(ptr)
    return tensor
end
