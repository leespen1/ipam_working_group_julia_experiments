"""
Retrieve the value wrapped by a Val type
"""
@inline function getVal(x::Val{T}) where {T} 
    return T
end

"""
Return the input variable.

Overloaded with getVal(x::Val{T}), so that the same code can use runtime or
compile time argument values.
"""
@inline function getVal(x)
    return x
end

