import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import datetime
import time
import json
import random
import argparse
import scorers
import optimizers
import generator
import evaluators
from get_utils import get_predictor, get_task_class, get_exs, is_generic_dataset

random.seed(42)

# 通用数据集的默认 prompt 模板
DEFAULT_GENERATE_PROMPT = """# Task
Describe the main object in the given image in comprehensive detail. Focus on distinctive visual features that help identify its specific category or species.

Please cover the following aspects if applicable:
- Overall shape and body structure
- Specific colors and patterns on different body parts (e.g., head, wings, body, legs)
- Textures (e.g., fur, feathers, scales, smooth, metallic)
- Unique distinctive markings

Ignore the background, lighting, and surrounding context. Focus solely on the object itself."""

DEFAULT_PREDICT_PROMPT = """# Task
Based on the visual description provided below, identify the specific category of the object.

# Prediction
Text: The image shows the following features: {{ text }}

Instruction: Compare the described features with the typical characteristics of possible categories. Identify the best match based on the distinctive attributes.

The answer is:"""


def load_prompts(task_name):
    """
    加载 prompt 模板
    
    对于通用数据集，使用默认模板
    对于原有任务，从文件加载
    """
    prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
    
    # 检查是否有对应的 prompt 文件
    generate_file = os.path.join(prompt_dir, f'{task_name}_generate.md')
    multi_file = os.path.join(prompt_dir, f'{task_name}_multi.md')
    
    if os.path.exists(generate_file) and os.path.exists(multi_file):
        # 使用已有的 prompt 文件
        candidates = [open(generate_file).read()]
        pred_prompt = open(multi_file).read()
        print(f"[load_prompts] 使用已有 prompt: {generate_file}")
    else:
        # 使用默认模板
        candidates = [DEFAULT_GENERATE_PROMPT]
        pred_prompt = DEFAULT_PREDICT_PROMPT
        print(f"[load_prompts] 使用默认 prompt 模板 (数据集: {task_name})")
    
    return candidates, pred_prompt


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')


def get_scorer(scorer):
    if scorer == 'compare':
        return scorers.CachedCompareScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='cub',
                        help='数据集名称: cub, dog, flower, car, pet, aircraft, eurosat, food, dtd, '
                             'caltech101, caltech256, sun397, imagenet_a, imagenet_r, imagenet_1k, '
                             'birdsnap, ucf, imagenet_sketch, imagenet_v2 或原有任务名')
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o', 'sglang_qwen'])
    parser.add_argument('--gradient_model', default='gemini')
    parser.add_argument('--data_dir', default='/datasets')
    parser.add_argument('--out_num', default='0')
    parser.add_argument('--max_threads', default=8, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_train', default=30, type=int, help='# instances per class')
    parser.add_argument('--n_val', default=30, type=int)
    parser.add_argument('--n_test', default=30, type=int)

    parser.add_argument('--minibatch_size', default=60, type=int, help='# total instances per minibatch')
    parser.add_argument('--n_gradients', default=4, type=int, help='# generated gradients per prompt')
    parser.add_argument('--errors_per_gradient', default=4, type=int,
                        help='# error examples used to generate one gradient')
    parser.add_argument('--gradients_per_error', default=1, type=int, help='# gradient reasons per error')
    parser.add_argument('--steps_per_gradient', default=1, type=int, help='# new prompts per gradient reason')
    parser.add_argument('--mc_samples_per_step', default=1, type=int, help='# synonyms')
    parser.add_argument('--max_expansion_factor', default=5, type=int, help='maximum # prompts after expansion')

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="compare", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--eval_budget', default=30, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)

    parser.add_argument('--reject_on_errors', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_seeds', default=10, type=int, help='# shuffle seeds in scorer')
    parser.add_argument('--test_eval', action='store_true', default=False)

    parser.add_argument("--train_ratio", type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    # 统一保存到 autosep/results/ 目录
    args.out = f"autosep/results/{args.out_num}_{args.task_name}/apo_multi_{args.task_name}_{args.out_num}.txt"
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    if args.evaluator != "bf1":
        configs['eval_budget'] = (configs['samples_per_eval'] * configs['eval_rounds']
                                  * configs['eval_prompts_per_round'])
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    task = get_task_class(args)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(configs)
    bf_eval = get_evaluator('bf')(configs)
    gpt4 = get_predictor(configs)
    gpt_generator = generator.AttrGredictor(configs)
    optimizer = optimizers.ProTeGi(configs, evaluator, scorer, args.max_threads, bf_eval)

    train_exs, val_exs, test_exs = get_exs(args, task)

    # 加载 prompt 模板（支持通用数据集和原有任务）
    candidates, pred_prompt = load_prompts(args.task_name)
    with open(args.out, 'a') as outf:
        outf.write(f'pred_prompt-------------------------\n')
        outf.write(f'{pred_prompt}\n\n')

    attribute_cache, test_attr_cache, pred_prompts = {}, {}, {}
    for prompt in candidates:
        attribute_cache[f'{prompt}'] = {}
        attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs,
                                                          attribute_cache, args.max_threads)

        test_attr_cache[f'{prompt}'] = {}
        if args.test_eval:
            test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs,
                                                              test_attr_cache, args.max_threads)
            pred_prompts[f'{prompt}'] = pred_prompt

    for round in tqdm(range(configs['rounds'] + 1)):
        print("STARTING ROUND ", round)
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
        start = time.time()

        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs, attribute_cache=attribute_cache)
            for prompt in candidates:
                if f'{prompt}' not in attribute_cache:
                    attribute_cache[f'{prompt}'] = {}
                    attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs,
                                                                      attribute_cache, args.max_threads)

        scores = optimizer.score_candidates(candidates, gpt4, train_exs, attribute_cache=attribute_cache)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        candidates = candidates[:configs['beam_size']]
        scores = scores[:configs['beam_size']]

        with open(args.out, 'a') as outf:
            outf.write(f'{time.time() - start}\n')
            for c in candidates:
                outf.write(json.dumps(c) + '\n')
            outf.write(f'{scores}\n')

        metrics = []
        for prompt in candidates:
            if f'{prompt}' not in test_attr_cache:
                test_attr_cache[f'{prompt}'] = {}
                if args.test_eval:
                    test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs,
                                                                      test_attr_cache, args.max_threads)
                    pred_prompts[f'{prompt}'] = pred_prompt

        if args.test_eval:
            for candidate, score in zip(candidates, scores):
                f1, texts, labels, preds, attr = task.evaluate(gpt4, candidate, test_exs, pred_prompts=pred_prompts,
                                                               attribute_cache=test_attr_cache, model_name=args.model)
                metrics.append(f1)
            with open(args.out, 'a') as outf:
                outf.write(f'{metrics}\n')

        # 保存属性缓存到 autosep/results/ 目录
        result_dir = f'autosep/results/{args.out_num}_{args.task_name}'
        os.makedirs(result_dir, exist_ok=True)
        with open(f'{result_dir}/{args.out_num}_train_attr.json', 'w') as json_file:
            json.dump(attribute_cache, json_file)
        with open(f'{result_dir}/{args.out_num}_test_attr.json', 'w') as json_file:
            json.dump(test_attr_cache, json_file)
        print(f"已保存属性缓存到: {result_dir}/")

    print("DONE!")
