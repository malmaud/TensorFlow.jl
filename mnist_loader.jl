using MLDatasets
import Random

mutable struct DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, Random.shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float32, batch_size, 784)
    y = zeros(Float32, batch_size, 10)
    for i in 1:batch_size
        data, label = MLDatasets.MNIST.traindata(loader.order[loader.cur_id])
        x[i, :] = reshape(data, (28*28))
        y[i, Int(label)+1] = 1.0
        loader.cur_id += 1
        if loader.cur_id > 60000
            loader.cur_id = 1
        end
    end
    x, y
end

function load_test_set(N=10000)
    x = zeros(Float32, N, 784)
    y = zeros(Float32, N, 10)
    for i in 1:N
        data, label = MLDatasets.MNIST.testdata(i)
        x[i, :] = reshape(data, (28*28))
        y[i, Int(label)+1] = 1.0
    end
    x,y
end
