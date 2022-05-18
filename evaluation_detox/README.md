For wieting similarity, you need to download the weights of the model [here](https://storage.yandexcloud.net/nlp/wieting_similarity_data.zip)

For Cola classifier for fluency, you need to download the weights of the model [here](https://drive.google.com/drive/folders/1p6_3lCbw3J0MhlidvKkRbG73qwmtWuRp).

To evaluate your predictions run: ```python evaluate.py --inputs <PATH_TO_INPUTS> --preds <PATH_TO_PREDS> --cola_classifier_path YOUR_COLA_PATH --wieting_model_path <YOUR_WIETING_MODEL_PATH> --wieting_tokenizer_path <YOUR_WIETING_TOKENIZER_PATH> --batch_size 32```
