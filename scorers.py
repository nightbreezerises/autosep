from collections import defaultdict
import numpy as np
from tqdm import tqdm
from autosep.llm_text_compare import select_k_from_n_excluding_i, predict_with_compare
from spo.llm_eval import select_k_from_n_excluding_i, prompt_spo_compare

# 注意: 已移除所有 ProcessPoolExecutor，改为纯顺序执行
# 原因: 本地 CUDA 模型无法在多进程间共享，fork 会导致死锁和僵尸进程


class Cached01Scorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompts=None, attribute_cache=None, agg='mean', max_threads=1,
                 model_name='gemini'):
        def compute_scores(prompts_exs):
            out_scores = {}
            for prompt, ex in tqdm(prompts_exs, desc='01 scorer'):
                if attribute_cache is not None:
                    pred_prompt = pred_prompts[f'{prompt}']
                    attr = attribute_cache[f'{prompt}'][f'{ex}']
                    pred = predictor.inference(pred_prompt, [ex['img_path']], attr)
                else:
                    pred = predictor.inference(prompt, [ex['img_path']])
                if pred == ex['label']:
                    out_scores[f'{ex}-{prompt}'] = 1
                else:
                    out_scores[f'{ex}-{prompt}'] = 0
            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))

        computed_scores = compute_scores(prompts_exs_to_compute)

        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [float(np.mean(cached_scores[prompt])) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)


class CachedCompareScorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompts=None, attribute_cache=None, agg='mean', max_threads=1,
                 model_name='gemini'):
        def compute_scores(prompts_exs):
            out_scores, prompt_scores = {}, {}
            for prompt, ex in prompts_exs:
                prompt_scores[f'{ex}-{prompt}'] = {}
                out_scores[f'{ex}-{prompt}'] = 0

            # 纯顺序执行，避免多进程死锁
            for prompt, true_ex in tqdm(prompts_exs, desc='compare scorer'):
                for false_ex in false_exs[f'{true_ex}']:
                    answer, true_ex, false_ex, prompt = predict_with_compare(true_ex, false_ex, prompt,
                                                                             attribute_cache[f'{prompt}'],
                                                                             model_name)
                    prompt_scores[f'{true_ex}-{prompt}'][f'{false_ex}'] = answer
                    out_scores[f'{true_ex}-{prompt}'] += answer

            return out_scores

        false_exs = {}
        for i in range(len(data)):
            false_idx = select_k_from_n_excluding_i(len(data), 2, i)  # compare
            false_exs[f'{data[i]}'] = [data[idx] for idx in false_idx]

        cached_scores = defaultdict(list)
        for prompt in prompts:
            prompts_exs_to_compute = []
            for ex in data:  # for ex, prompt in [(ex, prompt) for ex in data]:
                if f'{ex}-{prompt}' in self.cache:
                    cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
                else:
                    prompts_exs_to_compute.append((prompt, ex))

            computed_scores = compute_scores(prompts_exs_to_compute)

            for prompt, ex in prompts_exs_to_compute:
                self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
                cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [float(np.mean(cached_scores[prompt])) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)


class CachedSPOScorer:

    def __init__(self, args):
        self.opt = args
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompts=None, attribute_cache=None, agg='mean', max_threads=1,
                 model_name='gemini'):
        def compute_scores(prompts_exs):
            out_scores = {}
            for pi, ex, idx in prompts_exs:
                out_scores[f'{ex}-{pi}-{idx}'] = 0

            # 纯顺序执行，避免多进程死锁
            for pos_idx, ex, neg_idx in tqdm(prompts_exs, desc='compare SPO scorer'):
                answer, ex, pos, neg = prompt_spo_compare(ex, pos_idx, neg_idx,
                                                          attribute_cache[f'{prompts[pos_idx]}'],
                                                          attribute_cache[f'{prompts[neg_idx]}'],
                                                          self.opt['task_name'], model_name)
                out_scores[f'{ex}-{pos}-{neg}'] += answer

            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex in data:
            prompts_exs_to_compute.append((0, ex, 1))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for p, ex, idx in prompts_exs_to_compute:
            cached_scores[prompts[0]].append(computed_scores[f'{ex}-{p}-{idx}'])
            cached_scores[prompts[1]].append(1 - computed_scores[f'{ex}-{p}-{idx}'])

        if agg == 'mean':
            return [float(np.mean(cached_scores[prompt])) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)
