import gc
import argparse
from metric_tools.style_transfer_accuracy import *
from metric_tools.content_similarity import *
from metric_tools.fluency import *
from metric_tools.joint_metrics import *
import numpy as np


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)

    parser.add_argument("--cola_classifier_path", 
                       default='style_transfer/cola_classifier'
                       )
    parser.add_argument("--wieting_model_path",
                        default='im.pt'
                        )
    parser.add_argument("--wieting_tokenizer_path",
                        default='sim.sp.30k.model'
                        )

    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--t1", default=75, type=float)
    parser.add_argument("--t2", default=70, type=float)
    parser.add_argument("--t3", default=12, type=float)
    
    parser.add_argument("--toxification", action='store_true')
    parser.add_argument("--task_name", choices=['jigsaw', 'yelp'], default='jigsaw')
    args = parser.parse_args()


    with open(args.inputs, 'r') as input_file, open(args.preds, 'r') as preds_file:
        inputs = input_file.readlines()
        preds = preds_file.readlines()
        
    # accuracy of style transfer
    accuracy_by_sent = classify_preds(args, preds)
    accuracy = np.mean(accuracy_by_sent)
    cleanup()
    
    # similarity
    bleu = calc_bleu(inputs, preds)
    emb_sim_stats = flair_sim(args, inputs, preds)
    emb_sim = emb_sim_stats.mean()
    cleanup()


    similarity_by_sent = wieting_sim(args, inputs, preds)
    avg_sim_by_sent = similarity_by_sent.mean()
    cleanup()
    
    # fluency
    char_ppl = calc_flair_ppl(preds)
    cleanup()
    
    token_ppl = calc_gpt_ppl(preds)
    cleanup()
    
    cola_stats = do_cola_eval(args, preds)
    cola_acc = sum(cola_stats) / len(preds)
    cleanup()
    
    # count metrics
    gm = get_gm(args, accuracy, emb_sim, char_ppl)
    joint = get_j(args, accuracy_by_sent, similarity_by_sent, cola_stats, preds)
    
    # write res to table
    if not os.path.exists('results.md'):
        with open('results.md', 'w') as f:
            f.writelines('| Model | ACC | EMB_SIM | SIM | CharPPL | TokenPPL | FL | GM | J | BLEU |\n')
            f.writelines('| ----- | --- | ------- | --- | ------- | -------- | -- | -- | - | ---- |\n')
            
    with open('results.md', 'a') as res_file:
        name = args.preds.split('/')[-1]
        res_file.writelines(f'{name}|{accuracy:.4f}|{emb_sim:.4f}|{avg_sim_by_sent:.4f}|{char_ppl:.4f}|'
                            f'{token_ppl:.4f}|{cola_acc:.4f}|{gm:.4f}|{joint:.4f}|{bleu:.4f}|\n')