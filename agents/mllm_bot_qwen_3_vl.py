import sys
import os
import warnings

# 确保导入正确的utils模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.util import encode_base64, prepare_qwen2_5_input, get_important_image_tokens, create_attention_mask

import torch
from os import path
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from agents.CFG import CFGLogits 
from agents.attention import qwen_modify, qwen_modify_with_importance
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.measure import block_reduce

# 限制图片最大尺寸
pre_define_max_size=1440

# 抑制transformers生成配置的警告
warnings.filterwarnings('ignore', message='.*do_sample.*temperature.*', category=UserWarning)

QWEN = {
    'Qwen3-VL-8B': 'Qwen/Qwen3-VL-8B-Instruct'
}

# Qwen3-VL 推荐的生成超参数
QWEN3_VL_GENERATION_CONFIG = {
    'do_sample': True,
    'top_p': 0.8,
    'top_k': 20,
    'temperature': 0.7,
    'repetition_penalty': 1.0,
}

ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t ' \
                     'know honestly. Don\'t imagine any contents that are not in the image.'

SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following qwen2_5 huggingface demo


def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n + n_addition_q):]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []

    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + 'Question: {}'.format(questions[-1])
    else:
        chat_log = chat_log[:-2]
    return chat_log


def trim_answer(answer):
    if isinstance(answer, list):
        return answer
    answer = answer.split('Question:')[0].replace('\n', ' ').strip()
    return answer


class MLLMBot:
    def __init__(self, model_tag, model_name, pai_enable_attn=False, device='cpu', device_id=0, bit8=False, max_answer_tokens=-1):
        self.model_tag = model_tag
        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens

        local_model_path_abs = "./models/Qwen"
        local_model_path = path.join(local_model_path_abs, QWEN[self.model_tag].split('/')[-1])

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(local_model_path)

        print("\n================= 模型初始化（MLLMBot - Qwen3-VL） =================")
        print(f"📌 模型标识（model_tag）: {model_tag}")
        print(f"📌 模型名称（model_name）: {model_name}")
        print(f"📁 本地模型路径: {local_model_path}")
        print(f"📁 图片最大尺寸: {pre_define_max_size} ，超出这个值将压缩")

        # ========== CPU ==========
        if device == 'cpu':
            self.device = 'cpu'
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_model_path,
                torch_dtype="auto"
            )
            dtype_used = "auto（CPU 默认）"
            print(f"🖥️ 设备: CPU")

        # ========== GPU ==========
        else:
            self.device = f'cuda:{device_id}'
            self.bit8 = bit8

            print(f"🖥️ 设备: GPU - {self.device}")
            print(f"🤖 使用 8bit 推理: {'是' if self.bit8 else '否'}")

            # 按官方示例：使用 bfloat16 或 8bit 量化
            if self.bit8:
                dtype_config = {"load_in_8bit": True}
                dtype_used = "int8（8bit 量化推理）"
            else:
                # Qwen3-VL 官方推荐使用 bfloat16
                dtype_config = {"torch_dtype": torch.bfloat16}
                dtype_used = "bfloat16（官方推荐）"

            print(f"🔍 使用数据类型: {dtype_used}")

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_model_path,
                device_map="auto",
                **dtype_config
            ).eval()

            # 开启梯度检查点
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ 已启用梯度检查点以节省显存")

        print(f"📁 local_model_path: {local_model_path}")
        print(f"🔢 当前使用的精度 dtype: {dtype_used}")
        print(f"🔧 最大生成长度 max_answer_tokens: {self.max_answer_tokens}")
        print("🚀 模型加载完成！")
        print("========================================================\n")
        
        # TODO超参数
        self.pai_enable_attn = pai_enable_attn   # 阶段一：是否增强图像注意力
        self.pai_alpha = 0.5           # 阶段一：增强系数 α
        self.pai_layers = (10, 28)     # 阶段一：层先验（深层更有效）
        self.pai_enable_cfg = False    # 阶段二：是否开启CFG logits精炼
        self.pai_gamma = 1.1           # 阶段二：γ 指导强度
        self.num_map = 0
        
    def __del__(self):
        """析构函数：清理GPU内存"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"清理MLLMBot内存时出错: {e}")
    
    def cleanup(self):
        """手动清理内存"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            torch.cuda.empty_cache()
            print("MLLMBot内存已清理")
        except Exception as e:
            print(f"清理MLLMBot内存时出错: {e}")
        
    def _get_model_device(self):
        try:
            return self.model.model.embed_tokens.weight.device
        except Exception:
            # 退化方案：取第一个参数所在设备或 self.device
            try:
                return next(self.model.parameters()).device
            except Exception:
                return torch.device(self.device)

    # # TODO 这里应该需要考虑chunk切分
    # def _resolve_img_token_span(self, messages, inputs):
    #     """返回(img_start_idx, img_end_idx)。
    #     启发式：缺少显式 image special token 时，近似把末尾 256 个 token 当作图像区域。
    #     若序列过短或无法解析，则返回 (None, None) 跳过注入。
    #     """
    #     try:
    #         input_ids = inputs.input_ids
    #         if input_ids is None:
    #             print(f'input_ids is None')
    #             return None, None
    #         seq_len = input_ids.shape[1]
    #         img_tokens = 256
    #         print(f'input_ids:{input_ids.shape}\nseq_len:{seq_len}')
    #         if seq_len <= img_tokens:
    #             print(f'seq_len <= img_tokens')
    #             return None, None
    #         img_start = seq_len - img_tokens
    #         img_end = seq_len
    #         print(f'img_start:{img_start}, img_end:{img_end}')
    #         return img_start, img_end
    #     except Exception as e:
    #         print(f"error return None None:{e}")
    #         return None, None


    def _resolve_img_token_span(self, messages, inputs):
        try:
            input_ids = inputs.input_ids
            if input_ids is None:
                print(f'input_ids is None')
                return None, None
            seq_len = input_ids.shape[1]
            # tokenizer 里有 special token 的映射
            tokenizer = self.processor.tokenizer
            vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
            image_pad_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
            vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
            print(f'input_ids:{input_ids.shape}\nseq_len:{seq_len}')
            input_ids_list = input_ids[0].tolist()
            if vision_start_id in input_ids_list and vision_end_id in input_ids_list:
                img_start = input_ids_list.index(vision_start_id)
                img_end   = input_ids_list.index(vision_end_id) + 1  # 包含 img_end
                print(f"找到 image token span: img_start={img_start}, img_end={img_end}")
                return img_start, img_end
            else:
                print("未找到 image token span")
                return None, None
        except Exception as e:
            print(f"error return None None:{e}")
            return None, None

    def _inject_qwen_pai_attention(self, img_start_idx, img_end_idx):
        if img_start_idx is None or img_end_idx is None:
            print('[ATTN] skip injection for Qwen3 (img span unresolved).')
            return
        print(f'[ATTN] inject Qwen3 attention layers {self.pai_layers} alpha={self.pai_alpha} span=({img_start_idx},{img_end_idx})')
        qwen_modify(self.model, self.pai_layers[0], self.pai_layers[1], True, self.pai_alpha, False, img_start_idx, img_end_idx)

    def _inject_qwen_pai_attention_with_importance(self, img_start_idx, img_end_idx, important_tokens_info):
        if img_start_idx is None or img_end_idx is None:
            print('[ATTN] skip injection for Qwen3 (img span unresolved).')
            return
        
        print(f'[ATTN] inject Qwen3 attention layers with importance weights {self.pai_layers} alpha={self.pai_alpha} span=({img_start_idx},{img_end_idx})')
        
        # 提取重要性权重信息
        importance_weights = important_tokens_info['weights']  # 所有图像token的权重
        important_indices = important_tokens_info['important_indices']  # 重要token的索引
        
        # 调用修改函数，传递重要性信息
        qwen_modify_with_importance(self.model, self.pai_layers[0], self.pai_layers[1], True, self.pai_alpha, False, img_start_idx, img_end_idx, importance_weights, important_indices)

    def get_name(self):
        return self.model_name
    
    def _resize_image_if_needed(self, image: Image.Image, max_size: int = pre_define_max_size) -> Image.Image:
        """
        如果图像尺寸超过max_size，按比例缩小以防止显存爆炸
        
        Args:
            image: PIL图像
            max_size: 最大边长（默认pre_define_max_size，预定义好）
            
        Returns:
            调整后的PIL图像
        """
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > max_size:
            # 计算缩放比例
            scale = max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            print(f"图像过大 ({width}x{height})，缩小到 ({new_width}x{new_height}) 以节省显存")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

    def __call_qwen3(self, raw_image, prompt, max_new_tokens=256):
        """
        Qwen3-VL 推理方法，基于官方示例实现
        """
        if isinstance(raw_image, Image.Image):
            raw_image = [raw_image]

        # 构建 content 列表
        content = []
        for img in raw_image:
            # 限制图像最大尺寸，防止超大图片导致显存爆炸
            img = self._resize_image_if_needed(img, max_size=pre_define_max_size)
            # 直接传 PIL Image 对象，而不是 base64 字符串
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        
        # 构造 messages（官方格式）
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        # 使用官方推荐的 apply_chat_template 方法准备输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到模型设备
        model_device = self._get_model_device()
        inputs = inputs.to(model_device)

        # TODO: 注意力增强功能暂时禁用，Qwen3-VL 需要适配
        # if self.pai_enable_attn:
        #     pass

        # 清理显存缓存
        torch.cuda.empty_cache()
        
        # 检查显存使用情况 - 基于剩余显存的清理策略
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            memory_free = memory_total - memory_allocated
            print(f"推理前显存: 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB, 剩余={memory_free:.2f}GB")
            
            # 如果剩余显存 < 12GB，触发清理（适配A6000 48GB和A800 80GB）
            if memory_free < 12:  
                print(f"警告: 剩余显存不足 ({memory_free:.2f}GB < 12GB)，强制清理...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # 再次检查
                memory_after_clear = torch.cuda.memory_allocated() / 1024**3
                memory_free_after = memory_total - memory_after_clear
                print(f"清理后剩余显存: {memory_free_after:.2f}GB")
                if memory_free_after < 10:
                    torch.cuda.reset_peak_memory_stats()
                    print(f"已重置峰值显存统计")
        
        with torch.no_grad():
            # 使用 Qwen3-VL 推荐的生成参数
            generation_config = QWEN3_VL_GENERATION_CONFIG.copy()
            generation_config.update({
                'max_new_tokens': max_new_tokens,
                'use_cache': True,
                'pad_token_id': self.processor.tokenizer.eos_token_id,
            })
            
            # 官方示例的生成方式
            generated_ids = self.model.generate(
                **inputs,
                **generation_config
            )
            
        # 按官方示例处理输出
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        reply = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # 清理临时变量
        del inputs, generated_ids, generated_ids_trimmed
        
        # 基于剩余显存决定是否清理
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_free = memory_total - memory_after
            print(f"推理后显存: 已分配={memory_after:.2f}GB, 已保留={memory_reserved:.2f}GB, 剩余={memory_free:.2f}GB")
            
            # 只在剩余显存 < 10GB 时才清理，否则保留缓存提升性能
            if memory_free < 10:
                torch.cuda.empty_cache()
                print(f"剩余显存不足10GB，已清理缓存")
                
                # 如果保留的显存过多且剩余不足10GB，重置峰值统计
                if memory_reserved > memory_free and memory_free < 10:
                    torch.cuda.reset_peak_memory_stats()
                    print(f"已重置峰值显存统计")
        
        # print(f"test MLLM answer after decode: {reply}")
        return reply

    def answer_chat_log(self, raw_image, chat_log, n_context=-1):
        # prepare the context for qwen3
        qwen3_prompt = '\n'.join([ANSWER_INSTRUCTION,
                                  get_chat_log(chat_log['questions'],chat_log['answers'],
                                               last_n=n_context), SUB_ANSWER_INSTRUCTION]
                                 )

        reply = self.__call_qwen3(raw_image, qwen3_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def tell_me_the_obj(self, raw_image, super_class, super_unit):
        std_prompt = f"Questions: What is the {super_unit} of the {super_class} in this photo? Answer:"
        # std_prompt = f"Questions: What is the name of the main object in this photo? Answer:"
        reply = self.__call_qwen3(raw_image, std_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def describe_attribute(self, raw_image, attr_prompt, max_new_tokens=256):
        # raw_image是Image.open之后的格式   
        reply = self.__call_qwen3(raw_image, attr_prompt, max_new_tokens)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply
    
    def compare_attention_enhancement(self, raw_image, attr_prompt, save_dir="./experiments/attention_comparison"):
        """
        对比注意力增强前后的效果
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 60)
        print("ATTENTION ENHANCEMENT COMPARISON")
        print("=" * 60)
        
        # 1. 运行未增强版本
        print("\n[1] Running WITHOUT attention enhancement...")
        original_attn = self.pai_enable_attn
        self.pai_enable_attn = False
        
        reply_no_enhance, _ = self.describe_attribute(raw_image, attr_prompt)
        print(f"Without enhancement: {reply_no_enhance}")
        
        # 2. 运行增强版本
        print("\n[2] Running WITH attention enhancement...")
        self.pai_enable_attn = True
        
        reply_with_enhance, _ = self.describe_attribute(raw_image, attr_prompt)
        print(f"With enhancement: {reply_with_enhance}")
        
        # 3. 恢复原始设置
        self.pai_enable_attn = original_attn
        
        # 4. 保存对比结果
        with open(os.path.join(save_dir, "comparison_results.txt"), "w", encoding="utf-8") as f:
            f.write("ATTENTION ENHANCEMENT COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Prompt: {attr_prompt}\n\n")
            f.write(f"Without enhancement: {reply_no_enhance}\n\n")
            f.write(f"With enhancement: {reply_with_enhance}\n\n")
            f.write(f"Enhancement layers: {self.pai_layers}\n")
            f.write(f"Alpha value: {self.pai_alpha}\n")
        
        print(f"\n[3] Comparison results saved to {save_dir}")
        print("=" * 60)
        
        return reply_no_enhance, reply_with_enhance

    def caption(self, raw_image):
        # standard way to caption an image in the qwen3 paper
        std_prompt = 'a photo of'
        reply = self.__call_qwen3(raw_image, std_prompt)
        reply = reply[0] if isinstance(reply, list) else reply
        reply = reply.replace('\n', ' ').strip()  # trim caption
        return reply

    def call_llm(self, prompts):
        prompts_temp = self.processor(None, prompts, return_tensors="pt")
        model_device = self._get_model_device()
        input_ids = prompts_temp['input_ids'].to(model_device)
        attention_mask = prompts_temp['attention_mask'].to(model_device)

        prompts_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=prompts_embeds,
                attention_mask=attention_mask)

        outputs = self.processor.decode(outputs[0], skip_special_tokens=True)
        return outputs
