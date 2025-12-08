import sys
import os

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from abc import ABC, abstractmethod
from liquid import Template
from tqdm import tqdm
import api_utils as utils

# 注意: 已移除 concurrent.futures，改为纯顺序执行
# 原因: 本地 CUDA 模型无法在多进程间共享，fork 会导致死锁和僵尸进程


class GPT4Generator(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def generate(self, ex, prompt):
        pass


class AttrGredictor(GPT4Generator):
    categories = ['No', 'Yes']

    def generate(self, prompt, ex=None):
        prompt = Template(prompt).render()
        if 'gpt4o' in self.opt['model']:
            response = utils.gpt4o(prompt, [ex['img_path']])[0]
        elif 'sglang' in self.opt['model']:
            response = utils.sglang_model(prompt, [ex['img_path']], model_name=self.opt['model'])[0]
        else:
            response = utils.google_gemini(prompt, [ex['img_path']])[0]

        if response is None:
            print(f"No attributes generated for {ex['id']}\t{ex['img_path']}")
            with open(self.opt['out'], 'a') as outf:
                outf.write(f"No attributes generated for {ex['id']}\t{ex['img_path']}\n")
        return response

    def generate_category(self, prompt, exs=None):
        prompt = Template(prompt).render()
        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, [ex['img_path'] for ex in exs])[0]
        else:
            response = utils.gpt4o(prompt, [ex['img_path'] for ex in exs])[0]
        if response is None:
            print(f"No attributes generated category")
        return response


def parallel_generate(generator, prompt, examples, attr_cache, max_threads):
    """
    生成属性描述（纯顺序执行）
    
    注意：已移除所有多进程逻辑，避免 CUDA 死锁和僵尸进程问题。
    """
    for ex in tqdm(examples, desc='Generating'):
        pred = generator.generate(prompt, ex)
        attr_cache[f'{prompt}'][f'{ex}'] = pred
    return attr_cache
