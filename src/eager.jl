mutable struct EagerContext
    ptr::Ptr{Cvoid}

    function EagerContext()
        options = @tfcall(:TFE_NewContextOptions, Ptr{Cvoid}, ())
        @tfcall(:TFE_ContextOptionsSetAsync, Cvoid, (Ptr{Cvoid}, Cuchar), options, 0)
        status = Status()
        context = @tfcall(:TFE_NewContext, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), options, status)
        check_status(status)
        this = new(context)
        # finalizer(this) do self
        #     @tfcall(:TFE_DeleteContext, Cvoid, (Ptr{Cvoid},), self.ptr)
        # end
        @tfcall(:TFE_DeleteContextOptions, Cvoid, (Ptr{Cvoid},), options)
        return this
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, c::EagerContext) = c.ptr

mutable struct TensorHandle
    ptr::Ptr{Cvoid}

    function TensorHandle(tensor)
        status = Status()
        ptr = @tfcall(:TFE_NewTensorHandle, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), tensor.ptr, status)
        check_status(status)
        this = new(ptr)
        # finalizer(this) do self
        #     @tfcall(:TFE_DeleteTensorHandle, Cvoid, (Ptr{Cvoid},), self.ptr)
        # end
        return this
    end

    function TensorHandle()
        return new()
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::TensorHandle) = h.ptr

function async_wait(ctx::EagerContext)
    status = Status()
    @tfcall(:TFE_ContextAsyncWait, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), ctx, status)
    check_status(status)
end

function device_name(h::TensorHandle)
    status = Status()
    c_name = @tfcall(:TFE_TensorHandleDeviceName, Cstring, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    return unsafe_string(c_name)
end

function data_type(h::TensorHandle)
    return @tfcall(:TFE_TensorHandleDataType, TF_DataType, (Ptr{Cvoid},), h)
end

function set_attr_type(op, attr_name, value)
    @tfcall(:TFE_OpSetAttrType, Cvoid, (Ptr{Cvoid}, Cstring, TF_DataType), op, attr_name, value)
end

function resolve(h::TensorHandle)
    status = Status()
    ptr = @tfcall(:TFE_TensorHandleResolve, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    tensor = RawTensor(ptr)
    return tensor
end

mutable struct EagerOp
    ptr::Ptr{Cvoid}

    function EagerOp(ctx::EagerContext, op_name)
        status = Status()
        ptr = @tfcall(:TFE_NewOp, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), ctx, op_name, status)
        check_status(status)
        this = new(ptr)
        # finalizer(this) do self
        #     @tfcall(:TFE_DeleteOp, Cvoid, (Ptr{Cvoid},), self)
        # end
        return this
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, op::EagerOp) = op.ptr

function add_input(op::EagerOp, h::TensorHandle)
    status = Status()
    @tfcall(:TFE_OpAddInput, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), op, h, status)
    check_status(status)
    return
end

function execute(op::EagerOp)
    handle = TensorHandle()
    num_ret = Cint(1)
    status = Status()
    @tfcall(:TFE_Execute, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cvoid}), op, Ref(handle.ptr), Ref(num_ret), status)
    check_status(status)
    return handle
end

function test_eager(ctx)
    h1 = TensorHandle(RawTensor([1,2]))
    h2 = TensorHandle(RawTensor([3,4]))
    op = EagerOp(ctx, "Add")
    add_input(op, h1)
    add_input(op, h2)
    dtype = data_type(h1)
    set_attr_type(op, "T", dtype)
    res = execute(op)
    return res
end
