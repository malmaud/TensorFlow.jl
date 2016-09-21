type PTBiterator
    data
    num_steps
end

Base.start(iter::PTBiterator) = 1

function Base.next(iter::PTBiterator, i)
        x = iter.data[:, (i-1)*iter.num_steps+1:i*iter.num_steps]
        y = iter.data[:, (i-1)*iter.num_steps+2:i*iter.num_steps+1]
        x, y
end

Base.done(iter::PTBiterator, i) =  i > iter.length(data)

function _read_words(filename)
    s = readstring(filename)
    replace(s,"\n", "<eos>") |> split
end

function _build_vocab(filename)
    words = _read_words(filename)
    word_to_id = build_rank_dict(words)
    return word_to_id
end

function _file_to_word_ids(filename, word_to_id)
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word ∈ keys(word_to_id)]
end

"""
Load PTB raw data from data directory `datapath`.

Reads PTB text files, converts strings to integer ids,
and performs mini-batching of the inputs.

The PTB dataset comes from Tomas Mikolov's webpage:

http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

Args:
datapath: string path to the directory where simple-examples.tgz has
been extracted.

Returns:
tuple (train_data, valid_data, test_data, vocabulary)
where each of the data objects can be passed to PTBiterator.
"""
function ptb_raw_data(datapath="")

    trainpath = joinpath(datapath, "ptb.train.txt")
    validpath = joinpath(datapath, "ptb.valid.txt")
    testpath = joinpath(datapath, "ptb.test.txt")

    word_to_id = _build_vocab(trainpath)
    train_data = _file_to_word_ids(trainpath, word_to_id)
    valid_data = _file_to_word_ids(validpath, word_to_id)
    test_data = _file_to_word_ids(testpath, word_to_id)
    vocabulary = length(word_to_id)
    return train_data, valid_data, test_data, vocabulary
end

"""Iterate on the raw PTB data.

This generates batch_size pointers into the raw PTB data, and allows
minibatch iteration along these pointers.

Args:
raw_data: one of the raw data outputs from ptb_raw_data.
batch_size: int, the batch size.
num_steps: int, the number of unrolls.

Yields:
Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
The second element of the tuple is the same data time-shifted to the
right by one.

Raises:
Error: if batch_size or num_steps are too high.
"""
function PTBiterator(raw_data, batch_size, num_steps)
    raw_data = Array(Int32,raw_data)

    data_len = length(raw_data)
    batch_len = data_len ÷ batch_size
    data = zeros(Int32,batch_size, batch_len)
    for i in 1:batch_size
        data[i,:] = raw_data[(batch_len*i+1):(batch_len*(i + 1))]
    end
    epoch_size = (batch_len - 1) ÷ num_steps
    epoch_size == 0 && error("epoch_size == 0, decrease batch_size or num_steps")

    PTBiterator(data,num_steps)
end


"""
    build_rank_dict(provider, f::Function = identity)
    build_rank_dictf(f::Function, provider)
Takes an iterable collection, i.e., an array or whatever and creates a `Dict` with pairs `k => rank`
where the keys `k` are the entries in `provider` and `rank` is based on the number of times `k` occured in `provider`

`f::Function(l::Vector{Pair{eltype(provider),Int}})` can optionally be supplied. `f` is a function applied to a Vector of pairs, sorted after number of occurances, in descending order. This can be used to filter some entries with to many/few occurances or truncate the length of the dict based on occurances etc. Example:
```
a = round(randn(10000));
build_rank_dict(a)
build_rank_dict(a, x-> filter(y->y.first>0,x))
build_rank_dictf(a) do x # This style is equivalent to the row above
    filter(y->y.first>0,x)
end

```
"""
function build_rank_dict(provider, f::Function = identity)
  d = Dict{eltype(provider),Int}()
  for element in provider
    d[element] = get(d, element, 0) + 1
  end
  a = sort(collect(d), by=x->-x.second) |> f
  return Dict(x.first => i for (i,x) in enumerate(a))
end

build_rank_dictf(f::Function, provider) = build_rank_dict(provider, f)
