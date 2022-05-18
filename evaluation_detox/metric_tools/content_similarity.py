import torch
import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from torch.nn.functional import cosine_similarity

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    from ..wieting_similarity import SimilarityEvaluator
else:
    from wieting_similarity import SimilarityEvaluator


def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    return float(bleu_sim / counter)


def flair_sim(args, inputs, preds):
    print('Calculating flair embeddings similarity')
    sim = 0
    batch_size = args.batch_size
    inp_embed = []
    pred_embed = []

    embedder = FlairEmbeddings('news-forward')

    for i in range(0, len(inputs), batch_size):
        inp_part = [Sentence(sent) for sent in inputs[i:i + batch_size]]
        pred_part = [Sentence(sent) for sent in preds[i:i + batch_size]]

        inp_part = embedder.embed(inp_part)
        pred_part = embedder.embed(pred_part)

        for j in range(batch_size):
            if ((i + j) < len(inputs)):
                inp_sent_vec = torch.zeros(2048).cuda()
                pred_sent_vec = torch.zeros(2048).cuda()

                for k in range(len(inp_part[j])):
                    inp_sent_vec += inp_part[j][k].embedding
                inp_embed.append(inp_sent_vec.cpu() / (k + 1))

                for k in range(len(pred_part[j])):
                    pred_sent_vec += pred_part[j][k].embedding
                pred_embed.append(pred_sent_vec.cpu() / (k + 1))

    emb_sim = cosine_similarity(torch.stack(inp_embed), torch.stack(pred_embed))

    return emb_sim


def wieting_sim(args, inputs, preds):
    assert len(inputs) == len(preds)
    print('Calculating similarity by Wieting subword-embedding SIM model')

    sim_model = SimilarityEvaluator(args.wieting_model_path, args.wieting_tokenizer_path)

    sim_scores = []

    for i in tqdm.tqdm(range(0, len(inputs), args.batch_size)):
        sim_scores.extend(
            sim_model.find_similarity(inputs[i:i + args.batch_size], preds[i:i + args.batch_size])
        )

    return np.array(sim_scores)
