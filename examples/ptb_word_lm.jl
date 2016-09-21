#=Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar xvf simple-examples.tgz

=#


using TensorFlow
using Distributions
include("reader.jl")

type PTBModel
    batch_size
    num_steps
    size
    vocab_size
    input_data
    targets
    initial_state
    final_state
    cost
    lr
    train_op
    new_lr
    lr_update
end

type Config
    init_scale::Float64
    learning_rate::Float64
    max_grad_norm::Int
    num_layers::Int
    num_steps::Int
    hidden_size::Int
    max_epoch::Int
    max_max_epoch::Int
    keep_prob::Float64
    lr_decay::Float64
    batch_size::Int
    vocab_size::Int
end
Config(;init_scale = 0.1,learning_rate = 1.0,max_grad_norm = 5,num_layers = 2,num_steps = 20,hidden_size = 200,max_epoch = 4,max_max_epoch = 13,keep_prob = 1.0,lr_decay = 0.5,batch_size = 20,vocab_size = 10000) = Config(init_scale,learning_rate,max_grad_norm,num_layers,num_steps,hidden_size,max_epoch,max_max_epoch,keep_prob,lr_decay,batch_size,vocab_size)

Base.show(c::Config) = println(@sprintf("init_scale: \t%f \nlearning_rate: \t%f \nmax_grad_norm: \t%f \nnum_layers: \t%f \nnum_steps: \t%f \nhidden_size: \t%f \nmax_epoch: \t%f \nmax_max_epoch: \t%f \nkeep_prob: \t%f \nlr_decay: \t%f \nbatch_size: \t%f \nvocab_size: \t%f \n",c.init_scale,c.learning_rate,c.max_grad_norm,c.num_layers,c.num_steps,c.hidden_size,c.max_epoch,c.max_max_epoch,c.keep_prob,c.lr_decay,c.batch_size,c.vocab_size))


smallConfig = Config(
init_scale = 0.1,
learning_rate = 1.0,
max_grad_norm = 5,
num_layers = 2,
num_steps = 20,
hidden_size = 200,
max_epoch = 4,
max_max_epoch = 13,
keep_prob = 1.0,
lr_decay = 0.5,
batch_size = 20,
vocab_size = 10000)


mediumConfig = Config(
init_scale = 0.05,
learning_rate = 1.0,
max_grad_norm = 5,
num_layers = 2,
num_steps = 35,
hidden_size = 650,
max_epoch = 6,
max_max_epoch = 39,
keep_prob = 0.5,
lr_decay = 0.8,
batch_size = 20,
vocab_size = 10000)


largeConfig = Config(
init_scale = 0.04,
learning_rate = 1.0,
max_grad_norm = 10,
num_layers = 2,
num_steps = 35,
hidden_size = 1500,
max_epoch = 14,
max_max_epoch = 55,
keep_prob = 0.35,
lr_decay = 1 / 1.15,
batch_size = 20,
vocab_size = 10000)


testConfig = Config(
init_scale = 0.1,
learning_rate = 1.0,
max_grad_norm = 1,
num_layers = 1,
num_steps = 2,
hidden_size = 2,
max_epoch = 1,
max_max_epoch = 1,
keep_prob = 1.0,
lr_decay = 0.5,
batch_size = 20,
vocab_size = 10000)



function PTBModel(is_training=true, config=smallConfig)

    batch_size = config.batch_size
    num_steps = config.num_steps
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    input_data = placeholder(Int32, shape=[batch_size, num_steps])
    targets = placeholder(Int32, shape=[batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = nn.rnn_cell.LSTMCell(hidden_size)# TODO: nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=true)
    if is_training && config.keep_prob < 1
        # TODO: lstm_cell = nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = config.keep_prob)
    end
    cell = nn.rnn_cell.BasicRNNCell(config.num_layers) # TODO: cell = nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=true)

    initial_state = nn.zero_state(cell, batch_size, Float32)

    embedding = get_variable("embedding", [vocab_size, hidden_size], Float32)
    inputs = nn.embedding_lookup(embedding, input_data)

    if is_training && config.keep_prob < 1
        inputs = nn.dropout(inputs, config.keep_prob)
    end

    inputs = TensorFlow.split(1, num_steps, inputs)
    outputs, state = nn.rnn(cell, inputs, initial_state=initial_state)

    output = reshape(concat(1, outputs), [-1, hidden_size])
    softmax_w = get_variable("softmax_w", [hidden_size, vocab_size], Float32)
    softmax_b = get_variable("softmax_b", [vocab_size], Float32)
    logits = output * softmax_w + softmax_b
    loss = nn.seq2seq.sequence_loss_by_example([logits],[reshape(targets, [-1])],[ones([batch_size * num_steps], Float32)])
    cost = reduce_sum(loss) / batch_size
    final_state = state

    is_training || return


    lr = Variable(0.0, trainable=false)
    tvars = trainable_variables()
    # grads, _ = clip_by_global_norm(gradients(cost,tvars), config.max_grad_norm) # TODO: use when implemented
    optimizer = train.GradientDescentOptimizer(lr)
    train_op = train.minimize(optimizer, cost)
    # train_op = optimizer.apply_gradients(zip(grads, tvars)) # TODO: use when implemented

    new_lr = placeholder(Float32, shape=[], name="new_learning_rate")
    lr_update = assign(lr, new_lr)

    PTBModel(batch_size,num_steps,hidden_size,vocab_size,input_data,targets,initial_state,final_state,cost,lr,train_op,new_lr,lr_update)

end

"""
run_epoch(session, model::PTBModel, data, eval_op, verbose=false)
Runs the model on the given data.
"""
function run_epoch(session, model::PTBModel, data, eval_op, verbose=false)
    epoch_size = ((length(data) ÷ model.batch_size) - 1) ÷ model.num_steps
    start_time = time()
    costs = 0.0
    iters = 0
    state = run(session,model.initial_state)
    for (step, (x,y)) in enumerate(PTBiterator(data, model.batch_size, model.num_steps))
        # x, y = xy
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = Dict()
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for (i, (c,h)) in enumerate(model.initial_state)
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        end
        cost, state, _ = run(session,fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        if verbose && step % (epoch_size ÷ 10) == 10
            println(@sprintf("%.3f perplexity: %.3f speed: %.0f wps",step/epoch_size,exp(costs/iters),iters*model.batch_size/(time() - start_time)))
        end
    end
    return exp(costs / iters)
end

assign_lr(model, session, lr_value) = run(session,model.lr_update,feed_dict=Dict(model.new_lr => lr_value))

function main(datapath, _config::Config)
    traindata, validdata, testdata, _ = ptb_raw_data(datapath)

    config = deepcopy(_config)
    eval_config = deepcopy(_config)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    initializer = Uniform(-config.init_scale,config.init_scale)
    variable_scope("model", reuse=false, initializer=initializer) do
        m = PTBModel(true, config)
    end
    variable_scope("model", reuse=true, initializer=initializer) do
        mvalid = PTBModel(false, config)
        mtest = PTBModel(false, eval_config)
    end

    run(session, initialize_all_variables())

    for i in 0:config.max_max_epoch-1
        lr_decay = config.lr_decay^max(i - config.max_epoch, 0.0)
        assign_lr(m,session, config.learning_rate * lr_decay)

        println("Epoch: %d Learning rate: %.3f" % (i + 1, run(session, m.lr)))
        train_perplexity = run_epoch(session, m, traindata, m.train_op,
        verbose=true)
        println("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, validdata, no_op())
        println("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    end

    test_perplexity = run_epoch(session, mtest, testdata, no_op())
    println("Test Perplexity: %.3f" % test_perplexity)
end


filename = datapath = "/work/fredrikb/deeplearning/simple-examples/data/"
config = smallConfig


PTBModel(true, config)

# TODO: make this work: main(datapath,config)
