import os
import numpy as np
import math
import torch
import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fairseq.models.roberta import RobertaModel
from flair.embeddings import FlairEmbeddings
from fairseq.data.data_utils import collate_tokens


def calc_flair_ppl(preds, aggregate=True):
    print('Calculating character-level perplexity')
    flair_ppl = []
    model = FlairEmbeddings('news-forward').lm

    for sent in preds:
        try:
            pp = model.calculate_perplexity(sent)
        except Exception as e:
            print(f'Got exception "{e}" when calculating flair perplexity for sentence "{sent}"')
            pp = model.calculate_perplexity(sent + '.')
        flair_ppl.append(pp)

    if aggregate:
        return np.mean(flair_ppl)
    return np.array(flair_ppl)


def calc_gpt_ppl(preds, aggregate=True):
    detokenize = lambda x: x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )",
                                                                                                                 ")").replace(
        "( ", "(")

    print('Calculating token-level perplexity')
    gpt_ppl = []

    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda()
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    gpt_model.eval()

    with torch.no_grad():
        for sent in tqdm.tqdm(preds):
            sent = detokenize(sent)
            if len(sent) == 1:
                sent = sent + '.'
            input_ids = gpt_tokenizer.encode(sent)
            inp = torch.tensor(input_ids).unsqueeze(0).cuda()

            try:
                result = gpt_model(inp, labels=inp, return_dict=True)
                loss = result.loss.item()
            except Exception as e:
                print(f'Got exception "{e}" when calculating gpt perplexity for sentence "{sent}" ({input_ids})')
                loss = 100

            gpt_ppl.append(100 if np.isnan(loss) else math.exp(loss))

    if aggregate:
        return np.mean(gpt_ppl)
    return np.array(gpt_ppl)


def do_cola_eval(args, preds, soft=False):
    print('Calculating CoLA acceptability stats')

    detokenize = lambda x: x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )",
                                                                                                                 ")").replace(
        "( ", "(")

    path_to_data = os.path.join(args.cola_classifier_path, 'cola-bin')

    cola_roberta = RobertaModel.from_pretrained(args.cola_classifier_path,
                                                checkpoint_file='checkpoint_best.pt',
                                                data_name_or_path=path_to_data
                                                )
    cola_roberta.eval()
    cola_roberta.cuda()

    cola_stats = []

    for i in tqdm.tqdm(range(0, len(preds), args.batch_size), total=len(preds) // args.batch_size):
        sentences = preds[i:i + args.batch_size]

        # detokenize and BPE encode input
        sentences = [cola_roberta.bpe.encode(detokenize(sent)) for sent in sentences]

        batch = collate_tokens(
            [cola_roberta.task.source_dictionary.encode_line("<s> " + sent + " </s>", append_eos=False)
             for sent in sentences],
            pad_idx=1
        )

        batch = batch[:, :512]

        with torch.no_grad():
            predictions = cola_roberta.predict('sentence_classification_head', batch.long())

        if soft:
            prediction_labels = torch.softmax(predictions, axis=1)[:, 1].cpu().numpy()
        else:
            prediction_labels = predictions.argmax(axis=1).cpu().numpy()
        # label 0 means acceptable. Need to inverse
        cola_stats.extend(list(1 - prediction_labels))

    return np.array(cola_stats)