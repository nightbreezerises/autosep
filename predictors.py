from abc import ABC, abstractmethod
from liquid import Template
from collections import Counter
import api_utils as utils


class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass


class TwoClassPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, attr, prompt, img_paths=None):
        prompt = Template(prompt).render(text=attr)
        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, img_paths, max_tokens=6, temperature=self.opt['temperature'])[0]
        else:
            response = utils.gpt4o(prompt, img_paths, max_tokens=6, n=1, temperature=self.opt['temperature'])[0]
        if response is None:
            print("No response received from the model.")
            return 1
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred

    def zero_shot_inference(self, prompt, img_paths=None):
        prompt = Template(prompt).render()
        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, img_paths, max_tokens=6, temperature=self.opt['temperature'])[0]
        else:
            response = utils.gpt4o(prompt, img_paths, max_tokens=6, n=1, temperature=self.opt['temperature'])[0]
        if response is None:
            print("No response received from the model.")
            return 1
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred

class ThreeClassPredictor(GPT4Predictor):

    def inference(self, prompt, img_paths=None, attr=None, few_shot_files=None, content=None):
        prompt = Template(prompt).render(text=attr)

        if 'gemini' in self.opt['model']:
            if few_shot_files != None:
                response = utils.google_gemini(prompt, img_paths, few_shot_files=few_shot_files,
                                               max_tokens=16, temperature=self.opt['temperature'])[0]
            else:
                response = utils.google_gemini(prompt, img_paths, max_tokens=16, temperature=self.opt['temperature'])[0]
        elif 'sglang' in self.opt['model']:
            response = utils.sglang_model(prompt, img_paths, max_tokens=16, temperature=self.opt['temperature'],
                                          model_name=self.opt['model'])[0]
        elif 'gpt4o' in self.opt['model']:
            response = utils.gpt4o(prompt, img_paths, max_tokens=16, n=1, temperature=self.opt['temperature'])[0]
        else:
            raise Exception(f"Unsupported model: {self.opt['model']}")
        if response is None:
            print("No response received from the model.")
            return None

        if 'A.' in response or '**A' in response or 'A\n' in response or 'A \n' in response or '(A)' in response or ': A' in response or response == 'A':
            pred = 0
        elif 'B.' in response or '**B' in response or 'B\n' in response or 'B \n' in response or '(B)' in response or ': B' in response or response == 'B':
            pred = 1
        elif 'C.' in response or '**C' in response or 'C\n' in response or 'C \n' in response or '(C)' in response or ': C' in response or response == 'C':
            pred = 2
        else:
            print(f"No valid response. {response}")
            return None

        return pred

    def inference_majority_vote(self, prompt, img_paths=None, attr=None, content=None, n_votes=5):
        preds = []
        for i in range(n_votes):
            single_pred = self.inference(prompt, img_paths, attr, content)
            if single_pred != None:
                preds.append(single_pred)
        if len(preds) > 0:
            vote_pred = Counter(preds).most_common(1)[0][0]
        else:
            vote_pred = None

        return vote_pred


class MultiClassPredictor(GPT4Predictor):
    """
    多分类预测器，支持任意数量的类别
    通过解析模型返回的类别索引或类别名称来确定预测结果
    """
    
    def inference(self, prompt, img_paths=None, attr=None, few_shot_files=None, content=None):
        """
        执行推理
        
        Args:
            prompt: 提示模板
            img_paths: 图片路径列表
            attr: 属性描述文本
            few_shot_files: few-shot 示例文件
            content: 额外内容
        
        Returns:
            预测的类别索引，如果无法解析则返回 None
        """
        prompt = Template(prompt).render(text=attr)
        
        if 'gemini' in self.opt['model']:
            if few_shot_files is not None:
                response = utils.google_gemini(
                    prompt, img_paths, few_shot_files=few_shot_files,
                    max_tokens=32, temperature=self.opt['temperature']
                )[0]
            else:
                response = utils.google_gemini(
                    prompt, img_paths, max_tokens=32, 
                    temperature=self.opt['temperature']
                )[0]
        elif 'sglang' in self.opt['model']:
            response = utils.sglang_model(
                prompt, img_paths, max_tokens=32, 
                temperature=self.opt['temperature'],
                model_name=self.opt['model']
            )[0]
        elif 'gpt4o' in self.opt['model']:
            response = utils.gpt4o(
                prompt, img_paths, max_tokens=32, n=1, 
                temperature=self.opt['temperature']
            )[0]
        else:
            raise Exception(f"Unsupported model: {self.opt['model']}")
        
        if response is None:
            print("No response received from the model.")
            return None
        
        # 尝试解析响应
        pred = self._parse_response(response)
        return pred
    
    def _parse_response(self, response):
        """
        解析模型响应，提取预测的类别索引
        
        支持的格式:
        - 直接数字: "42", "0", "123"
        - 带标签: "Class: 42", "Label: 5", "Answer: 10"
        - 字母选项: "A", "B", "C", ... (转换为 0, 1, 2, ...)
        - 带括号: "(A)", "(42)", "[B]"
        """
        import re
        
        response = response.strip()
        
        # 1. 尝试直接提取数字
        # 匹配 "42", "Class: 42", "Label: 5", "Answer: 10" 等
        number_patterns = [
            r'^\s*(\d+)\s*$',                    # 纯数字
            r'(?:class|label|answer|prediction)[:\s]*(\d+)',  # 带标签
            r'\((\d+)\)',                         # 括号内数字
            r'\[(\d+)\]',                         # 方括号内数字
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # 2. 尝试解析字母选项 (A=0, B=1, C=2, ...)
        letter_patterns = [
            r'^\s*([A-Z])\s*$',                   # 纯字母
            r'^\s*([A-Z])\.',                     # A. B. C.
            r'\*\*([A-Z])',                       # **A**
            r'\(([A-Z])\)',                       # (A)
            r'\[([A-Z])\]',                       # [A]
            r':\s*([A-Z])\s*$',                   # : A
        ]
        
        for pattern in letter_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                return ord(letter) - ord('A')
        
        # 3. 如果都失败，打印警告
        print(f"[MultiClassPredictor] 无法解析响应: {response[:100]}")
        return None
    
    def inference_majority_vote(self, prompt, img_paths=None, attr=None, content=None, n_votes=5):
        """多数投票推理"""
        preds = []
        for i in range(n_votes):
            single_pred = self.inference(prompt, img_paths, attr, content)
            if single_pred is not None:
                preds.append(single_pred)
        
        if len(preds) > 0:
            vote_pred = Counter(preds).most_common(1)[0][0]
        else:
            vote_pred = None
        
        return vote_pred
