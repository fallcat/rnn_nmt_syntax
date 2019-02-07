from model.seq2seq import *
# from train import *

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import random



def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH, span_size=SPAN_SIZE):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = tuple([torch.tensor([[SOS_token]], device=device) for i in range(span_size)])  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(int((max_length + 1) / span_size)):
            #             print("input[0]", decoder_input[0].shape)
            #             print("hidden[0]", decoder_hidden[0].shape)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            #             topv, topi = decoder_output.topk(1, dim=1)
            topi = [EOS_token] * SPAN_SIZE
            for si in range(span_size):
                topv, topi[si] = decoder_output[si].topk(1)
            decoder_input = tuple([topi[si].squeeze().detach() for si in range(span_size)])

            #             decoder_input = tuple(topi.squeeze().detach())
            for si in range(span_size):
                if di * span_size + si < max_length and topi[si].item() != EOS_token:
                    decoded_words.append(output_lang.index2word[topi[si].item()])
                else:
                    decoded_words.append('<EOS>')
                    break

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(input_lang, output_lang, encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluate_and_show_attention(input_lang, output_lang, encoder, decoder, input_sentence):
    output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

