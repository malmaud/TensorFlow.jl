@enum PlacementPolicy begin
    PLACEMENT_EXPLICIT=0
    PLACEMENT_WARN=1
    PLACEMENT_SILENT=2
    PLACEMENT_SILENT_FOR_INT32=3
end

mutable struct EagerContext <: Context
    ptr::Ptr{Cvoid}
end

function EagerContext(;async=false, placement_policy=nothing)
    # For some reason, `get_all_op_list` 
    # has to be called before :TFE_Execute or else tf
    # crashes. Maybe something about TF_GetAllOpList is causing the tf 
    # library to enter a bad state.
    get_all_op_list() 

    options = @tfcall(:TFE_NewContextOptions, Ptr{Cvoid}, ())
    @tfcall(:TFE_ContextOptionsSetAsync, Cvoid, (Ptr{Cvoid}, Cuchar), options, async)
    if placement_policy !== nothing
        @tfcall(:TFE_ContextOptionsSetDevicePlacementPolicy, Cvoid, (Ptr{Cvoid}, Int), options, placement_policy)
    end

    status = Status()
    context = @tfcall(:TFE_NewContext, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), options, status)
    check_status(status)
    this = EagerContext(context)
    finalizer(this) do self
        @tfcall(:TFE_DeleteContext, Cvoid, (Ptr{Cvoid},), self.ptr)
    end
    @tfcall(:TFE_DeleteContextOptions, Cvoid, (Ptr{Cvoid},), options)
    return this
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, c::EagerContext) = c.ptr

function DeviceList(ctx::EagerContext)
    status = Status()
    ptr = @tfcall(:TFE_ContextListDevices, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), ctx, status)
    check_status(status)
    count = @tfcall(:TF_DeviceListCount, Cint, (Ptr{Cvoid},), ptr)
    this = DeviceList(ptr, count)
    return this
end

mutable struct EagerTensor <: AbstractTensor{Any}
    ptr::Ptr{Cvoid}
    EagerTensor(ptr::Ptr) = new(ptr)
end

function EagerTensor(tensor::RawTensor)
    status = Status()
    ptr = @tfcall(:TFE_NewTensorHandle, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), tensor.ptr, status)
    check_status(status)
    this = EagerTensor(ptr)
    finalizer(this) do self
        @tfcall(:TFE_DeleteTensorHandle, Cvoid, (Ptr{Cvoid},), self.ptr)
    end
    return this
end

EagerTensor(value) = EagerTensor(RawTensor(value))
Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::EagerTensor) = h.ptr

function async_wait(ctx::EagerContext)
    status = Status()
    @tfcall(:TFE_ContextAsyncWait, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), ctx, status)
    check_status(status)
end

function device_name(h::EagerTensor)
    status = Status()
    c_name = @tfcall(:TFE_TensorHandleDeviceName, Cstring, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    return unsafe_string(c_name)
end

function backing_device_name(h::EagerTensor)
    status = Status()
    c_name = @tfcall(:TFE_TensorHandleBackingDeviceName, Cstring, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    return unsafe_string(c_name)
end


function data_type(h::EagerTensor)
    return @tfcall(:TFE_TensorHandleDataType, TF_DataType, (Ptr{Cvoid},), h) |> tf_to_jl_type
end

Base.eltype(h::EagerTensor)  = data_type(h)

function resolve(h::EagerTensor)
    status = Status()
    ptr = @tfcall(:TFE_TensorHandleResolve, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    tensor = RawTensor(ptr)
    return tensor
end

Base.convert(::Type{Array}, h::EagerTensor) = convert(Array, resolve(h))

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
    ctx = get_eager_context()
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

function add_input(op::EagerOp, h::EagerTensor)
    status = Status()
    @tfcall(:TFE_OpAddInput, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), op, h, status)
    check_status(status)
    return
end

function add_input(op::EagerOp, hs::Vector{EagerTensor})
    for h in hs
        add_input(op, h)
    end
end

function execute(op::EagerOp)
    op_desc = get_op_def(op.op_name)
    n_outputs = length(op_desc.output_arg)
    handles = [EagerTensor(C_NULL) for _ in 1:n_outputs]
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
    list = Int64.(list)
    @tfcall(:TFE_OpSetAttrIntList, Cvoid, (Ptr{Cvoid}, Cstring, Ptr{Int64}, Cint), op, attr_name, list, length(list))
end

function set_attr_list(op::EagerOp, attr_name, list::Vector{<:AbstractFloat})
    list = Float32.(list)
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

function clear_caches(ctx::EagerContext)
    @tfcall(:TFE_ContextClearCaches, Cvoid, (Ptr{Cvoid},), ctx)
end

function num_dims(h::EagerTensor)
    status = Status()
    res = @tfcall(:TFE_TensorHandleNumDims, Cint, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    Int(res)
end

function num_elements(h::EagerTensor)
    status = Status()
    res = @tfcall(:TFE_TensorHandleNumElements, Int64, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    Int(res)
end


function dim(h::EagerTensor, dim_index)
    status = Status()
    res = @tfcall(:TFE_TensorHandleDim, Int64, (Ptr{Cvoid}, Cint, Ptr{Cvoid}), h, dim_index-1, status)
    check_status(status)
    Int(res)
end

function copy_sharing_tensor(h::EagerTensor)
    status = Status()
    res = EagerTensor()
    res.ptr = @tfcall(:TFE_TensorHandleCopySharingTensor, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), h, status)
    check_status(status)
    return res
end

function copy_to_device(ctx::EagerContext, h::EagerTensor, device_name)
    status = Status()
    res = EagerTensor()
    res.ptr = @tfcall(:TFE_TensorHandleCopyToDevice, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Cstring, Ptr{Cvoid}), h, ctx, device_name, status)
    check_status(status)
    return res
end

copy_to_device(h, device_name) = copy_to_device(get_eager_context(), h, device_name)

function set_device(op::EagerOp, device_name)
    status = Status()
    @tfcall(:TFE_OpSetDevice, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Ptr{Cvoid}), op, device_name, status)
    check_status(status)
end

function get_device(op::EagerOp)
    status = Status()
    str = @tfcall(:TFE_OpGetDevice, Cstring, (Ptr{Cvoid}, Ptr{Cvoid}), op, status)
    return String(str)
end


"""
    set_xla_compilation(op::EagerOp, enable::Bool)

When 'enable' is 'true', and if TensorFlow is build with XLA support, a subsequent
'execute' call on `op` will run the op via XLA.

If the library is not built with XLA support, this call is a no-op.
"""
function set_xla_compilation(op::EagerOp, enable)
    @tfcall(:TFE_OpSetXLACompilation, Ptr{Cvoid}, (Ptr{Cvoid}, Cuchar), op, enable)
end

Base.convert(::Type{EagerTensor}, h::EagerTensor) = h
Base.convert(::Type{EagerTensor}, h) = constant(h)

function item(t::EagerTensor)
    x = convert(Array, t)
    if length(x) != 1
        throw(DimensionMismatch("item can only be called on scalar tensors"))
    end
    return x[1]
end

Base.length(t::EagerTensor) = item(Ops.size(t))

Base.IteratorEltype(::Type{EagerTensor}) = Base.EltypeUnknown() # temp hack
Base.eltype(::Type{EagerTensor}) = Any
Base.collect(t::EagerTensor) = Array(t)
Base.iterate(t::EagerTensor, args...) = iterate(Array(t), args...)
Base.zero(t::AbstractTensor) = Ops.zeros_like(t)
Base.ones(t::AbstractTensor) = Ops.ones_like(t)

function Base.:*(t1::EagerTensor, t2::Number)
    return t1 .* t2
end

function Base.:*(t1::Number, t2::EagerTensor)
    return t1 .* t2
end

function inplace_sub(x, y)
    i = cast(constant(0:(item(size(x,0))-1)), Int32)
    Ops.inplace_sub(x, i, y)
end

struct ExecutionMode <: Context
    eager::Bool
end

ExecutionMode(;eager=true) = ExecutionMode(eager)

function enable_eager_execution()
    # context = Context()
    # context.attrs["eager"] = true
    # push!(global_context, context)
    push!(global_context, ExecutionMode(eager=true))
    return nothing
end

function in_eager_mode()
    return context_value(ExecutionMode).eager
end

function get_eager_context()
    ctx = context_value(EagerContext)
    if ctx === nothing
        ctx = EagerContext()
        push!(global_context, ctx)
    end
    return ctx
end
