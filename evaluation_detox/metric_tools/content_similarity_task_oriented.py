from .style_transfer_accuracy_formality import load_model, prepare_target_label, classify_texts


def evaluate_meaning(
    original_texts,
    rewritten_texts,
    model_name='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
    target_label='entailment',
    bidirectional=True,
    batch_size=32,
    verbose=False,
    aggregation='prod',
    model=None,
    tokenizer=None,
):
    model, tokenizer = load_model(model_name, model, tokenizer)
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        original_texts, rewritten_texts,
        model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    if bidirectional:
        reverse_scores = classify_texts(
            rewritten_texts, original_texts,
            model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
        )
        if aggregation == 'prod':
            scores = reverse_scores * scores
        elif aggregation == 'mean':
            scores = (reverse_scores + scores) / 2
        elif aggregation == 'f1':
            scores = 2 * reverse_scores * scores / (reverse_scores + scores)
        else:
            raise ValueError('aggregation should be one of "mean", "prod", "f1"')
    return scores