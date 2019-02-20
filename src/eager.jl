mutable struct EagerContext
    ptr::Ptr{Cvoid}

    function EagerContext()
        # For some reason, this has to be called before :TFE_Execute or else tf
        # crashes. Maybe something about TF_GetAllOpList is causing the tf 
        # library to enter a bad state.
        get_all_op_list() 

        options = @tfcall(:TFE_NewContextOptions, Ptr{Cvoid}, ())
        @tfcall(:TFE_ContextOptionsSetAsync, Cvoid, (Ptr{Cvoid}, Cuchar), options, 0)
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

eager_ctx = nothing #EagerContext()
eager_mode = true

Base.unsafe_convert(::Type{Ptr{Cvoid}}, c::EagerContext) = c.ptr

function DeviceList(ctx::EagerContext)
    status = Status()
    ptr = @tfcall(:TFE_ContextListDevices, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), ctx, status)
    check_status(status)
    count = @tfcall(:TF_DeviceListCount, Cint, (Ptr{Cvoid},), ptr)
    this = new(ptr, count)
    return this
end

mutable struct TensorHandle <: AbstractTensor{Any}
    ptr::Ptr{Cvoid}

    function TensorHandle(tensor::RawTensor)
        status = Status()
        ptr = @tfcall(:TFE_NewTensorHandle, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), tensor.ptr, status)
        check_status(status)
        this = new(ptr)
        finalizer(this) do self
            @tfcall(:TFE_DeleteTensorHandle, Cvoid, (Ptr{Cvoid},), self.ptr)
        end
        return this
    end

    function TensorHandle()
        return new()
    end
end

EagerTensor(value) = TensorHandle(RawTensor(value))
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
    return @tfcall(:TFE_TensorHandleDataType, TF_DataType, (Ptr{Cvoid},), h) |> tf_to_jl_type
end

Base.eltype(h::TensorHandle)  = data_type(h)

function resolve(h::TensorHandle)
    status = Status()
    ptr = @tfcall(:TFE_TensorHandleResolve, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    tensor = RawTensor(ptr)
    return tensor
end

function Base.Array(h::TensorHandle)
    convert(Array, resolve(h))
end

mutable struct EagerOp
    ptr::Ptr{Cvoid}
    op_name::String
end

function EagerOp(ctx::EagerContext, op_name)
    status = Status()
    ptr = @tfcall(:TFE_NewOp, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), ctx, op_name, status)
    check_status(status)
    this = EagerOp(ptr, String(op_name))
    finalizer(this) do self
        @tfcall(:TFE_DeleteOp, Cvoid, (Ptr{Cvoid},), self)
    end
    return this
end

function EagerOp(op_name)
    global eager_ctx
    if eager_ctx === nothing
        eager_ctx = EagerContext()
    end
    ctx = eager_ctx
    status = Status()
    ptr = @tfcall(:TFE_NewOp, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), ctx, op_name, status)
    check_status(status)
    this = EagerOp(ptr, String(op_name))
    finalizer(this) do self
        @tfcall(:TFE_DeleteOp, Cvoid, (Ptr{Cvoid},), self)
    end
    return this
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, op::EagerOp) = op.ptr

function add_input(op::EagerOp, h::TensorHandle)
    status = Status()
    @tfcall(:TFE_OpAddInput, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), op, h, status)
    check_status(status)
    return
end

function execute(op::EagerOp)
    op_desc = get_op_def(op.op_name)
    n_outputs = length(op_desc.output_arg)
    handles = [TensorHandle() for _ in 1:n_outputs]
    ptrs = [Ptr{Cvoid}(0) for _ in 1:n_outputs]
    num_ret = Ref{Cint}(n_outputs)
    status = Status()
    @tfcall(:TFE_Execute, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cvoid}), op, ptrs, num_ret, status)
    check_status(status)
    for i in 1:n_outputs
        handles[i].ptr = ptrs[i]
    end
    return handles
end

function test_eager()
    ctx = EagerContext()
    h1 = TensorHandle(RawTensor([1,2]))
    h2 = TensorHandle(RawTensor([3,4]))
    op = EagerOp(ctx, "Add")
    add_input(op, h1)
    add_input(op, h2)
    dtype = data_type(h1)
    op["T"] = dtype
    res = execute(op)
    return res[1]
    # return resolve(res[1])
end

function setindex!(op::EagerOp, tensor::RawTensor, attr_name)
    status = Status()
    @tfcall(:TFE_OpSetAttrTensor, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Ptr{Cvoid}), op, attr_name, tensor, status)
    check_status(status)
end

function setindex!(op::EagerOp, dtype::DataType, attr_name)
    @tfcall(:TFE_OpSetAttrType, Cvoid, (Ptr{Cvoid}, Cstring, TF_DataType), op, attr_name, dtype |> jl_to_df_type)
end

function setindex!(op::EagerOp, value::Integer, attr_name)
    value = Int64(value)
    @tfcall(:TFE_OpSetAttrInt, Cvoid, (Ptr{Cvoid}, Cstring, Int64), op, attr_name, value)
end

function setindex!(op::EagerOp, value::Bool, attr_name)
    @tfcall(:TFE_OpSetAttrBool, Cvoid, (Ptr{Cvoid}, Cstring, Cuchar), op, attr_name, value)
end

function setindex!(op::EagerOp, value::AbstractFloat, attr_name)
    value = Float32(value)
    @tfcall(:TFE_OpSetAttrFloat, Cvoid, (Ptr{Cvoid}, Cstring, Cfloat), op, attr_name, value)
end

function setindex!(op::EagerOp, value::AbstractString, attr_name)
    value = String(value)
    @tfcall(:TFE_OpSetAttrString, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Cint), op, attr_name, Vector{UInt8}(value), sizeof(value))
end

function setindex!(op::EagerOp, value::Vector, attr_name)
    set_attr_list(op, attr_name, value)
end

function set_attr_list(op::EagerOp, attr_name, list::Vector{<:Integer})
    list = Int64[Int64(x) for x in list]
    @tfcall(:TFE_OpSetAttrIntList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Int64}, Cint), op, attr_name, list, length(list))
end

function set_attr_list(op::EagerOp, attr_name, list::Vector{<:AbstractFloat})
    list = Float32[Float32(x) for x in list]
    @tfcall(:TFE_OpSetAttrFloatList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Float32}, Cint), op, attr_name, list, length(list))
end

function set_attr_list(op::EagerOp, attr_name, list::Vector{<:DataType})
    list = map(jl_to_df_type, list)
    @tfcall(:TFE_OpSetAttrTypeList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Cint), op, attr_name, list, length(list))
end

function set_attr_shape_list(op::EagerOp, attr_name, list::Vector)
    dims = Vector{Int64}[]
    for shape in list
        push!(dims, Int64[shape...])
    end
    @tfcall(:TFE_OpSetAttrShapeList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Ptr{Int64}}, Ptr{Cint}, Cint),
        op,
        attr_name,
        dims,
        Cint[length(x) for x in dims],
        length(dims))
end
