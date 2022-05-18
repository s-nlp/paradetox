from transformers import RobertaTokenizer, RobertaForSequenceClassification
import tqdm
from torch.nn.utils.rnn import pad_sequence


def classify_preds(args, preds):
    print('Calculating style of predictions')
    results = []

    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    for i in tqdm.tqdm(range(0, len(preds), args.batch_size)):
        batch = tokenizer(preds[i:i + args.batch_size], return_tensors='pt', padding=True)
        result = model(**batch)['logits'].argmax(1).float().data.tolist()
        results.extend([1 - item for item in result])

    return results