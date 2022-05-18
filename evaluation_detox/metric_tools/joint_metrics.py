def get_gm(args, accuracy, emb_sim, char_ppl):

    return (max(100 * accuracy - args.t1, 0) * max(100 * emb_sim - args.t2, 0) * max(args.t3 - char_ppl, 0)) ** (1/3)


def get_j(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds):

    return sum(accuracy_by_sent * similarity_by_sent * cola_stats) / len(preds)