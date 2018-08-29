module seq2seq

export
sequence_loss_by_example,
sequence_loss

import ..nn: tf
import .tf: @not_implemented, @op

@op function sequence_loss_by_example(logits, targets, weights;
                                  average_across_timesteps=true,
                                  softmax_loss_function=nothing,
                                  name=nothing)
    local score
    tf.with_op_name(name, "SequenceLossByExample") do
        scores = tf.Tensor[]
        for (logit, target, weight) in zip(logits, targets, weights)
            if softmax_loss_function === nothing
                push!(scores, tf.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target) .* weight)
            else
                push!(scores, softmax_loss_function(logit, target) .* weight)
            end
        end
        score = tf.add_n(scores)
        if average_across_timesteps
            weight = tf.add_n(weights)
            score = score/weight
        end
    end
    score
end

@op function sequence_loss(logits, targets, weights,
                       average_across_timesteps=true,
                       average_across_batch=true,
                       softmax_loss_function=nothing,
                       name=nothing)
    local score
    tf.with_op_name(name, "SequenceLoss") do
        score = tf.reduce_sum(sequence_loss_by_example(logits, tagets, weight, average_across_timesteps=average_across_timesteps, softmax_loss_function=softmass_loss_function))
        if average_across_batch
            batch_size = get_shape(logits[1]).dims[1]
            score = score/(eltype(score)(batch_size))
        end
    end
    score
end

@not_implemented function basic_rnn_seq2seq()
end

@not_implemented function one2many_rnn_seq2seq()
end

@not_implemented function tied_rnn_seq2seq()
end

@not_implemented function model_with_buckets()
end


function rnn_decoder(decoder_inputs, initial_state, cell; scope="", loop_function=nothing)
    state = initial_state
    outputs = tf.Tensor[]
    prev = nothing
    for (i, input) in enumerate(decoder_inputs)
        tf.variable_scope(scope, reuse=i>1) do
            if prev !== nothing && loop_function !== nothing
                tf.variable_scope("loop_function", reuse=true) do
                    input = loop_function(prev, i)
                end
            end
            output, state = cell(input, state)
            push!(outputs, output)
            prev = output
        end
    end
    outputs, state
end

end
