import argparse
import json
import re
import difflib
from dvc_eval import eval_soda, eval_dvc
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from bert_nli.bert_nli import BertNLIModel
import os
import moxing as mox


def merge_similar_sentences(data):
    if not data: return data
    merged_data = []
    current_sentence = data[0]["sentence"]
    current_timestamp = data[0]["timestamp"]
    for i in range(1, len(data)):
        next_sentence = data[i]["sentence"]
        next_timestamp = data[i]["timestamp"]
        if difflib.SequenceMatcher(None, current_sentence, next_sentence).ratio() > 0.98 and -1 <= next_timestamp[0] - current_timestamp[1] <= 1:
            current_timestamp = [current_timestamp[0], next_timestamp[1]]
        else:
            merged_data.append({"sentence": current_sentence, "timestamp": current_timestamp})
            current_sentence = next_sentence
            current_timestamp = next_timestamp
    merged_data.append({"sentence": current_sentence, "timestamp": current_timestamp})
    return merged_data


def grounding_metrics(evals):
    ious = []
    for sample in evals:
        meta = sample['meta']
        duration = meta['duration']
        token = meta['token']

        gt = sample['answer']
        se = []
        for k, v in token.items():
            if k in gt:
                se.append(v)
        if len(se) == 2:
            s, e = se[0], se[1]
            if s > e:
                s, e = e, s
            s = s / duration
            e = e / duration
        else:
            continue

        prediction = sample['pred']
        matches = re.search(r"(\d{2,4}) (to|and) (\d{2,4})", prediction)
        if not matches:
            iou = 0
        else:
            from_number = float(matches.group(1)) / (10 ** (len(matches.group(1))))
            to_number = float(matches.group(3)) / (10 ** (len(matches.group(3))))

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            iou = intersection / (union + 1e-5)
        ious.append(iou)

    l = len(ious)
    assert l > 0
    metrics = {
        "mIoU": sum(ious) / l * 100
    }
    for m in [0.3, 0.5, 0.7]:
        metrics[f"R1@{m}"] = sum(iou >= m for iou in ious) / l * 100
    print(metrics)


def captioning_metrics(evals):
    pred = {}
    gt = {}
    for sample in evals:
        id = sample['id']

        pred[id] = sample['pred']
        gt[id] = sample['answer']

    scorers = [
        (Bleu(4), "Bleu_4"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    task_metric = dict()
    for scorer, method in scorers:
        print("computing %s score" % (scorer.method()))
        score, scores = scorer.compute_score(gt, pred)
        task_metric[method] = score
    print(task_metric)


def accuracy_nli_metrics(evals):
    bert_nli = BertNLIModel('/cache/model_zoo/bert_nli/bert-base.state_dict')
    bert_nli.eval()
    corrects = 0
    for sample in evals:
        pred = sample['pred']
        question = sample['question']
        answer = sample['answer']

        qp = question + "?" + pred if "?" not in question else question + " " + pred
        qa = question + "?" + answer if "?" not in question else question + " " + answer

        try:
            labels, _ = bert_nli([(qa, qp)])  # todo: parallel
        except:
            labels = ['contradiction']

        if labels[0] in ['entail']:
            corrects += 1
        elif labels[0] in ['contradiction', 'neutral']:
            pass
        else:
            raise NotImplementedError('Unrecognized label: {}!'.format(labels[0]))
    acc = corrects / len(evals)
    print("Accuracy in NLI: {}".format(acc))


def accuracy_gpt_metrics(evals, args):
    gpt_messages = []
    for sample in evals:
        pred = sample['pred']
        question = sample['question']
        answer = sample['answer']

        gpt_message = dict(
            prompt_new="Please evaluate the following video-based question-answer pair:\n\n"
                       f"Question: {question}\n"
                       f"Correct Answer: {answer}\n"
                       f"Predicted Answer: {pred}\n\n"
                       "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                       "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                       "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                       "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
            system_prompt="You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                          "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                          "------"
                          "##INSTRUCTIONS: "
                          "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                          "- Consider synonyms or paraphrases as valid matches.\n"
                          "- Evaluate the correctness of the prediction compared to the answer.",
            id=sample['id']
        )
        gpt_messages.append(gpt_message)
    tgt_dir_cloud = os.path.dirname(args.eval_file)
    tgt_path_cloud = os.path.join(tgt_dir_cloud, 'gpt.json')
    with open(tgt_path_cloud, 'w') as f:
        for gpt_message in gpt_messages:
            f.write(json.dumps(gpt_message) + "\n")
    exp = os.path.basename(tgt_dir_cloud)
    tgt_path_s3 = "s3://{}/nieming/exps-bak/LLaMA-VID/{}/{}.json".format(args.bucket_name, exp, exp)
    mox.file.copy_parallel(tgt_path_cloud, tgt_path_s3)
    print("gpt file has been uploaded to s3")


def metrics(evals, **kwargs):
    grd_evals = []
    cap_evals = []
    rsn_evals = []
    for sample in evals:
        if 'grd' in sample['task']:
            grd_evals.append(sample)
        elif 'cap' in sample['task']:
            cap_evals.append(sample)
        elif 'rsn' in sample['task']:
            rsn_evals.append(sample)
    print("====================== temporal grounding tasks =====================")
    grounding_metrics(grd_evals)
    print("====================== temporal captioning tasks =====================")
    captioning_metrics(cap_evals)
    print("====================== temporal reasoning tasks: cap =====================")
    captioning_metrics(rsn_evals)
    # print("====================== temporal reasoning tasks: acc-nli =====================")
    # accuracy_nli_metrics(rsn_evals)
    print("====================== temporal reasoning tasks: acc-gpt =====================")
    accuracy_gpt_metrics(rsn_evals, kwargs['args'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default=None)
    parser.add_argument('--bucket_name', type=str, default="bucket-9329")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    evals = []
    with open(args.eval_file, 'r') as f:
        for line in f.readlines():
            tmp = json.loads(line)
            evals.append(tmp)
    print("====================== Calculating metrics =====================")
    metrics(evals, args=args)
