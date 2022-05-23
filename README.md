# ParaDetox: Detoxification with Parallel Data

This repository contains information about Paradetox dataset -- the first parallel corpus for the detoxification task -- as well as models and evaluation methodology for the detoxification of English texts. The original paper ["ParaDetox: Detoxification with Parallel Data"](https://aclanthology.org/2022.acl-long.469/) was presented at ACL 2022 main conference.

## ParaDetox Collection Pipeline

The ParaDetox Dataset collection was done via [Yandex.Toloka](https://toloka.yandex.com/) crowdsource platform. The collection was done in three steps:
* *Task 1:* **Generation of Paraphrases**: The first crowdsourcing task asks users to eliminate toxicity in a given sentence while keeping the content.
* *Task 2:* **Content Preservation Check**:  We show users the generated paraphrases along with their original variants and ask them to indicate if they have close meanings.
* *Task 3:* **Toxicity Check**: Finally, we check if the workers succeeded in removing toxicity.

The whole pipeline is illustrated on this schema:
![](https://github.com/skoltech-nlp/paradetox/blob/main/img/generation_pipeline_blue.jpg)

All these steps were done to ensure high quality of the data and make the process of collection automated. For more details please refer to the original paper.

## ParaDetox Dataset
As a result,  we get paraphrases for 11,939 toxic sentences (on average 1.66 paraphrases per sentence), 19,766 paraphrases total. The whole dataset can be found [here](https://github.com/skoltech-nlp/paradetox/blob/main/paradetox/paradetox.tsv). The examples of samples from ParaDetox Dataset:

![](https://github.com/skoltech-nlp/paradetox/blob/main/img/paraphrase_example.png)

ParaDetox dataset can be also obtained via HuggingFace🤗 [repo](https://huggingface.co/datasets/SkolkovoInstitute/paradetox). In addition to all ParaDetox dataset, we also make public [samples](https://github.com/skoltech-nlp/paradetox/blob/main/paradetox/paradetox_cannot_rewrite.tsv) that were marked by annotators as "cannot rewrite" in *Task 1* of the crowdsource pipeline.

# Detoxification evaluation

The automatic evaluation of the model were produced based on three parameters:
* *style transfer accuracy* (**STA**): percentage of nontoxic outputs identified by a style classifier. We pretrained toxicity classifier on Jigsaw data and put it online in HuggingFace🤗 [repo](https://huggingface.co/SkolkovoInstitute/roberta_toxicity_classifier).
* *content preservation* (**SIM**): cosine similarity between the embeddings of the original text and the output computed with the model of [Wieting et al. (2019)](https://aclanthology.org/P19-1427/).
* *fluency* (**FL**): percentage of fluent sentences identified by a RoBERTa-based classifier of linguistic acceptability trained on the [CoLA dataset](https://nyu-mll.github.io/CoLA/). 

All code used for our experiments to evluate different detoxifcation models can be run via Colab notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xTqbx7IPF8bVL2bDCfQSDarA43mIPefE?usp=sharing)

## Detoxification model
**New SOTA** for detoxification task -- BART (base) model trained on ParaDetox dataset -- we released online in HuggingFace🤗 repository [here](https://huggingface.co/SkolkovoInstitute/bart-base-detox).

You can also check out our [demo](https://detoxifier.nlp.zhores.net/junction/) and telegram [bot](https://t.me/rudetoxifierbot).

## Citation

```
@inproceedings{logacheva-etal-2022-paradetox,
    title = "{P}ara{D}etox: Detoxification with Parallel Data",
    author = "Logacheva, Varvara  and
      Dementieva, Daryna  and
      Ustyantsev, Sergey  and
      Moskovskiy, Daniil  and
      Dale, David  and
      Krotova, Irina  and
      Semenov, Nikita  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.469",
    pages = "6804--6818",
    abstract = "We present a novel pipeline for the collection of parallel data for the detoxification task. We collect non-toxic paraphrases for over 10,000 English toxic sentences. We also show that this pipeline can be used to distill a large existing corpus of paraphrases to get toxic-neutral sentence pairs. We release two parallel corpora which can be used for the training of detoxification models. To the best of our knowledge, these are the first parallel datasets for this task.We describe our pipeline in detail to make it fast to set up for a new language or domain, thus contributing to faster and easier development of new parallel resources.We train several detoxification models on the collected data and compare them with several baselines and state-of-the-art unsupervised approaches. We conduct both automatic and manual evaluations. All models trained on parallel data outperform the state-of-the-art unsupervised models by a large margin. This suggests that our novel datasets can boost the performance of detoxification systems.",
}
```

## Contacts

If you find some issue, do not hesitate to add it to [Github Issues](https://github.com/skoltech-nlp/paradetox/issues).

For any questions, please contact: Daryna Dementieva (daryna.dementieva@skoltech.ru)
