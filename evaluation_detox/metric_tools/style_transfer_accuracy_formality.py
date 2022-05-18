import torch
import numpy as np
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


def load_model(model_name=None, model=None, tokenizer=None, model_class=AutoModelForSequenceClassification):
    if model is None:
        if model_name is None:
            raise ValueError('Either model or model_name should be provided')
        model = model_class.from_pretrained(model_name)
        if torch.cuda.is_available():
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError('Either tokenizer or model_name should be provided')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_target_label(model, target_label):
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(f'target_label "{target_label}" is not in model labels or ids: {model.config.id2label}.')
    return target_label


def classify_texts(texts, second_texts=None, model_name=None, target_label=None, batch_size=32, verbose=False, model=None, tokenizer=None):
    model, tokenizer = load_model(model_name, model, tokenizer)
    target_label = prepare_target_label(model, target_label)
    res = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        inputs = [texts[i:i+batch_size]]
        if second_texts is not None:
            inputs.append(second_texts[i:i+batch_size])
        inputs = tokenizer(*inputs, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            preds = torch.softmax(model(**inputs).logits, -1)[:, target_label].cpu().numpy()
        res.append(preds)
    return np.concatenate(res)


def evaluate_formality(
    texts,
    model_name='cointegrated/roberta-base-formality',
    target_label=1,  # 1 is formal, 0 is informal
    batch_size=32,
    verbose=False,
    model=None,
    tokenizer=None,
):
    model, tokenizer = load_model(model_name, model, tokenizer)
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        texts,
        model=model, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose, target_label=target_label
    )
    return scores