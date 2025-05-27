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


"""
Like Iterators.partition, but instead of choosing the number of elements per
partition, choose the number of partitions, so they get divided up easily.
"""
function chunked_partition(collection, n_chunks)
    Base.require_one_based_indexing(collection)
    partitions = typeof(collection)[]
    n_els = length(collection)
    n_el_per_chunk, remainder = divrem(n_els, n_chunks)
    start = 1
    for i in 1:n_chunks
        stop = start+n_el_per_chunk-1
        stop += (i <= remainder) ? 1 : 0
        push!(partitions, collection[start:stop])
        start = stop+1
    end

    return partitions
end
