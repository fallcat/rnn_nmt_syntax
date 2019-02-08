from model.seq2seq import *
from model.utils import *
from evaluate import *

from torch import optim


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = tuple([torch.tensor([[SOS_token]], device=device) for i in range(span_size)])
#     print("len decoder input", len(decoder_input))

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(int((target_length+1)/span_size)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = []
            for si in range(span_size):
                if di*span_size+si < target_length:
                    loss += criterion(decoder_output[si], target_tensor[di*span_size+si])
                    decoder_input.append(target_tensor[di*span_size+si])
                else:
                    break
            decoder_input = tuple(decoder_input)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(int((target_length+1)/2)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topi = [EOS_token] * SPAN_SIZE
            for si in range(span_size):
                topv, topi[si] = decoder_output[si].topk(1)
            decoder_input = tuple([topi[si].squeeze().detach() for si in range(span_size)])  # detach from history as input

            for si in range(span_size):
                if di*span_size+si < target_length and decoder_input[si].item() != EOS_token:
                    loss += criterion(decoder_output[si], target_tensor[di*span_size+si])
                else:
                    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


if __name__ == "__main__":
    hidden_size = 256

    input_lang, output_lang, pairs, vocab = prepare_data('en', 'de', True)

    print(len(pairs))
    print(pairs[0])
    print(pairs[1])
    print(random.choice(pairs))

    encoder1 = EncoderRNN(len(vocab), hidden_size).to(device)
    attn_decoder1 = AttnKspanDecoderRNN(hidden_size, len(vocab), dropout_p=0.1).to(device)

    train_iters(encoder1, attn_decoder1, 1000, print_every=5000)
    # trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

    start = time.time()
    evaluate_randomly(input_lang, output_lang, encoder1, attn_decoder1, pairs)
    print("Time elapsed:", time.time() - start)

    # output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, "je suis trop froid .")
    # plt.matshow(attentions.numpy())
    #
    # evaluate_and_show_attention(input_lang, output_lang, encoder1, attn_decoder1, "elle a cinq ans de moins que moi .")
    #
    # evaluate_and_show_attention(input_lang, output_lang, encoder1, attn_decoder1, "elle est trop petit .")
    #
    # evaluate_and_show_attention(input_lang, output_lang, encoder1, attn_decoder1, "je ne crains pas de mourir .")
    #
    # evaluate_and_show_attention(input_lang, output_lang, encoder1, attn_decoder1, "c est un jeune directeur plein de talent .")