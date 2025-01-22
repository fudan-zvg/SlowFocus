import argparse
import torch

from slowfocus.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from slowfocus.conversation import conv_templates, SeparatorStyle
from slowfocus.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os

import math
from tqdm import tqdm
from decord import VideoReader, cpu
import re
import numpy as np


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def convert(duration, x, disc):
    x = x /duration * disc
    x = str(min(round(x), disc - 1))
    while len(x) < len(str(disc)) - 1:
        x = "0" + x
    return x


def merge(nums1, nums2):
    m = len(nums1)
    n = len(nums2)
    k = m + n - 1
    new_nums = nums1 + [0] * n
    lfreq_token_ids = []
    hfreq_token_ids = []
    while m > 0 and n > 0:
        if nums1[m - 1] > nums2[n - 1]:
            new_nums[k] = nums1[m - 1]
            m -= 1
            lfreq_token_ids.append(k)
        else:
            new_nums[k] = nums2[n - 1]
            n -= 1
            hfreq_token_ids.append(k)
        k -= 1
    new_nums[:n] = nums2[:n]
    while m > 0:
        lfreq_token_ids.append(m - 1)
        m -= 1
    while n > 0:
        hfreq_token_ids.append(n - 1)
        n -= 1
    lfreq_token_ids.reverse()
    hfreq_token_ids.reverse()
    return new_nums, lfreq_token_ids, hfreq_token_ids


def segment_search(string):
    segment = []
    while True:
        matches = re.search(r"(\d{2,4}) (to|and) (\d{2,4})", string)
        if matches is not None:
            s = matches.group(1)
            e = matches.group(3)
            if int(s) < int(e):
                segment.append((s, e))
            string = string[matches.span()[1]:]
        else:
            break
    return segment


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frame_num", type=int, default=25)
    parser.add_argument("--is_search", type=int, default=0)
    parser.add_argument("--search", type=str, default=None)
    parser.add_argument("--search_sampling", type=int, default=20)
    parser.add_argument("--is_prompt", type=int, default=0)
    parser.add_argument("--grd_discretization", type=int, default=100)
    parser.add_argument("--convert", type=bool, default=True)

    return parser.parse_args()


def video_sampling(video, sampling_fps, max_frame_num, sampling='sparse', sampling_num=0, starts=None, ends=None):
    if sampling in ['sparse', 'dense_v0']:
        fps = min(max(round(video.get_avg_fps() / sampling_fps), 1), len(video))
        if len(video) / fps > max_frame_num:
            fps = round(len(video) / max_frame_num)
        frame_idx = [i for i in range(0, len(video), fps)]
        lfreq_token_ids = torch.LongTensor([0])
        hfreq_token_ids = torch.LongTensor([0])

    if sampling in ['dense_v0', 'dense_v1']:
        frame_idx_ext = []
        for start, end in zip(starts, ends):
            start_idx = round(start * len(video))
            end_idx = min(round(end * len(video)), len(video) - 1)
            if end_idx - start_idx > sampling_num:
                sampling_idx = np.linspace(start_idx, end_idx, num=sampling_num)
                frame_idx_ext.extend(sampling_idx.astype(int).tolist())
            else:
                frame_idx_ext.extend(list(range(start_idx, end_idx + 1)))

        if sampling == 'dense_v0':
            frame_idx, lfreq_token_ids, hfreq_token_ids = merge(frame_idx, frame_idx_ext)
            lfreq_token_ids = torch.LongTensor(lfreq_token_ids)
            hfreq_token_ids = torch.LongTensor(hfreq_token_ids)
        elif sampling == 'dense_v1':
            frame_idx = frame_idx_ext
        else:
            raise NotImplementedError("Unsupported sampling strategy!")
    temporal_pos = torch.LongTensor(frame_idx) / len(video)  # detailed discretizaiton is included in model
    frames = video.get_batch(frame_idx).asnumpy()
    return frames, temporal_pos, lfreq_token_ids, hfreq_token_ids


def free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids, hfreq_token_ids, question, args):
    images = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().cuda()
    temporal_pos = temporal_pos.to(images)
    images = [images]  # align to possible multi-prompts
    temporal_pos = [temporal_pos]
    lfreq_token_ids = [lfreq_token_ids]
    hfreq_token_ids = [hfreq_token_ids]

    # remove redundant tokens
    question = question.strip("<video>\n")
    question = question.strip("<image>\n")

    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    cur_prompt = question
    with torch.inference_mode():
        model.update_prompt([[cur_prompt]])
        output_ids = model.generate(
            input_ids,
            images=images,
            temporal_pos=temporal_pos,
            lfreq_token_ids=lfreq_token_ids,
            hfreq_token_ids=hfreq_token_ids,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def run_inference(args):
    """
    Run inference on temporal bench.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gts = json.load(file)
    gts = get_chunk(gts, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    search_task = ['rsn']
    prompt_task = ['cap']

    for index, sample in enumerate(tqdm(gts)):
        if any(task in sample['task'] for task in search_task) and args.is_search:
            sample_set = inference_search(sample, model, image_processor, tokenizer, args)
        elif any(task in sample['task'] for task in prompt_task) and args.is_prompt:
            sample_set = inference_prompt(sample, model, image_processor, tokenizer, args)
        else:
            sample_set = inference_naive(sample, model, image_processor, tokenizer, args)
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()


def inference_prompt(sample, model, image_processor, tokenizer, args):
    video_file = sample['video']
    video_name = os.path.join('fineaction', 'videos', video_file)
    question = sample['question']
    id = sample['id']
    answer = sample['answer']
    meta = sample['meta']
    task = sample['task']

    sample_set = {'id': id, 'question': question, 'answer': answer, 'meta': meta, 'task': task}

    # step 0: convert question
    replace_set = []
    target_idx = []
    for k, v in meta['token'].items():
        replace_set.append((k, convert(meta['duration'], v, args.grd_discretization)))
    for x1, x2 in replace_set:
        if x1 in question:
            if args.convert:
                question = question.replace(x1, x2)
            target_idx.append(x2)

    # Check if the video exists
    video_path = os.path.join(args.video_dir, video_name)
    try:
        video = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print("loading error", e)
        print("Error file", video_path)

    # step 1: process prompt localization
    assert len(target_idx) in [0, 2], "prompt segment should be 0 or 2"
    if len(target_idx) > 0:
        segment = [sorted(target_idx)]
    else:
        segment = []

    # step 2: additional clues
    if len(segment) == 0:
        images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num)
    else:
        from_numbers = []
        to_numbers = []
        for seg in segment:
            from_number = float(seg[0]) / (10 ** (len(seg[0])))
            to_number = float(seg[1]) / (10 ** (len(seg[1])))
            from_numbers.append(from_number)
            to_numbers.append(to_number)

        images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num,
                                                                                sampling=args.search,
                                                                                sampling_num=args.search_sampling,
                                                                                starts=from_numbers,
                                                                                ends=to_numbers)

    outputs = free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids, hfreq_token_ids, question, args)
    sample_set['pred'] = outputs
    return sample_set


def inference_prompt_search(sample, model, image_processor, tokenizer, args):
    video_file = sample['video']
    video_name = os.path.join('fineaction', 'videos', video_file)
    question = sample['question']
    id = sample['id']
    answer = sample['answer']
    meta = sample['meta']
    task = sample['task']

    sample_set = {'id': id, 'question': question, 'answer': answer, 'meta': meta, 'task': task}

    # step 0: convert question
    replace_set = []
    target_idx = []
    for k, v in meta['token'].items():
        replace_set.append((k, convert(meta['duration'], v, args.grd_discretization)))
    for x1, x2 in replace_set:
        if x1 in question:
            if args.convert:
                question = question.replace(x1, x2)
            target_idx.append(x2)

    # Check if the video exists
    video_path = os.path.join(args.video_dir, video_name)
    try:
        video = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print("loading error", e)
        print("Error file", video_path)

    # step 1: process prompt localization
    assert len(target_idx) in [0, 2], "prompt segment should be 0 or 2"
    if len(target_idx) > 0:
        segment = [sorted(target_idx)]
    else:
        segment = []

    # step 2: additional clues
    if len(segment) == 0:
        images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num)
    else:
        from_numbers = []
        to_numbers = []
        for seg in segment:
            from_number = float(seg[0]) / (10 ** (len(seg[0])))
            to_number = float(seg[1]) / (10 ** (len(seg[1])))
            from_numbers.append(from_number)
            to_numbers.append(to_number)

        images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num,
                                                                                sampling=args.search,
                                                                                sampling_num=args.search_sampling,
                                                                                starts=from_numbers,
                                                                                ends=to_numbers)

    # step 3: prepare input for grounding
    grounding_question = "<video>\nCould you provide the time interval helpful to reason {}".format(
        question)

    outputs = free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids,
                                  hfreq_token_ids, grounding_question, args)

    # step 4: process grounding output
    segment = segment_search(outputs)

    # step 5: additional clues
    if len(segment) == 0:
        pass
    else:
        from_numbers = []
        to_numbers = []
        added_prompts = []
        for seg in segment:
            from_number = float(seg[0]) / (10 ** (len(seg[0])))
            to_number = float(seg[1]) / (10 ** (len(seg[1])))
            from_numbers.append(from_number)
            to_numbers.append(to_number)
            added_prompts.append("from {} to {}".format(seg[0], seg[1]))
        added_prompts = ", ".join(added_prompts)
        added_prompts += '.'
        added_prompts = " Additional temporal information to focus on:" + added_prompts
        question += added_prompts

        images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num,
                                                                                sampling=args.search,
                                                                                sampling_num=args.search_sampling,
                                                                                starts=from_numbers,
                                                                                ends=to_numbers)

    outputs = free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids, hfreq_token_ids, question, args)
    sample_set['pred'] = outputs
    return sample_set


def inference_search(sample, model, image_processor, tokenizer, args):
    video_file = sample['video']
    video_name = os.path.join('fineaction', 'videos', video_file)
    question = sample['question']
    id = sample['id']
    answer = sample['answer']
    meta = sample['meta']
    task = sample['task']

    sample_set = {'id': id, 'question': question, 'answer': answer, 'meta': meta, 'task': task}

    # step 0: convert question
    if args.convert:
        replace_set = []
        for k, v in meta['token'].items():
            replace_set.append((k, convert(meta['duration'], v, args.grd_discretization)))
        for x1, x2 in replace_set:
            if x1 in question:
                question = question.replace(x1, x2)

    # Check if the video exists
    video_path = os.path.join(args.video_dir, video_name)
    try:
        video = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print("loading error", e)
        print("Error file", video_path)

    # step 1: prepare input for grounding
    images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num)
    grounding_question = "<video>\nCould you provide the time interval helpful to reason {}".format(
        question)

    outputs = free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids, hfreq_token_ids, grounding_question, args)

    # step 2: process grounding output
    # todo: need to be comparitable
    segment = segment_search(outputs)

    # step 3: additional clues
    if len(segment) == 0:
        pass
    else:
        from_numbers = []
        to_numbers = []
        added_prompts = []
        for seg in segment:
            from_number = float(seg[0]) / (10 ** (len(seg[0])))
            to_number = float(seg[1]) / (10 ** (len(seg[1])))
            from_numbers.append(from_number)
            to_numbers.append(to_number)
            added_prompts.append("from {} to {}".format(seg[0], seg[1]))
        added_prompts = ", ".join(added_prompts)
        added_prompts += '.'
        added_prompts = " Additional temporal information to focus on:" + added_prompts
        question += added_prompts

        images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num,
                                                                                sampling=args.search,
                                                                                sampling_num=args.search_sampling,
                                                                                starts=from_numbers,
                                                                                ends=to_numbers)

    outputs = free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids, hfreq_token_ids, question, args)
    sample_set['pred'] = outputs
    return sample_set


def inference_naive(sample, model, image_processor, tokenizer, args):
    video_file = sample['video']
    video_name = os.path.join('fineaction', 'videos', video_file)
    question = sample['question']
    id = sample['id']
    answer = sample['answer']
    meta = sample['meta']
    task = sample['task']

    sample_set = {'id': id, 'question': question, 'answer': answer, 'meta': meta, 'task': task}

    # step 0: convert question
    if args.convert:
        replace_set = []
        for k, v in meta['token'].items():
            replace_set.append((k, convert(meta['duration'], v, args.grd_discretization)))
        for x1, x2 in replace_set:
            if x1 in question:
                question = question.replace(x1, x2)

    # Check if the video exists
    video_path = os.path.join(args.video_dir, video_name)
    try:
        video = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print("loading error", e)
        print("Error file", video_path)

    # step 1: prepare input
    images, temporal_pos, lfreq_token_ids, hfreq_token_ids = video_sampling(video, args.fps, args.max_frame_num)

    outputs = free_form_inference(model, image_processor, tokenizer, images, temporal_pos, lfreq_token_ids, hfreq_token_ids, question, args)
    sample_set['pred'] = outputs
    return sample_set


if __name__ == "__main__":
    args = parse_args()
    args.is_search = True if args.is_search == 1 else False
    args.is_prompt = True if args.is_prompt == 1 else False
    run_inference(args)