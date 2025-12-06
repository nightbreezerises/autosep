import os
import sys

sys.path.append("./sglang/python")
sys.path.append("./../sglang/python")
import re
import time
import base64
import config
import string
import PIL.Image
import openai
from openai import OpenAI, BadRequestError
import pathlib

# 全局变量用于缓存 MLLMBot 实例，避免重复加载模型
_mllm_bot_instance = None
_current_model_name = None  # 记录当前加载的模型名称

# 支持的模型配置
SUPPORTED_MODELS = {
    'Qwen2.5-VL-7B-Instruct': {
        'module': 'agents.mllm_bot_qwen_2_5_vl',
        'model_tag': 'Qwen2.5-VL-7B',
        'model_name': 'Qwen2.5-VL-7B-Instruct',
    },
    'Qwen3-VL-8B-Instruct': {
        'module': 'agents.mllm_bot_qwen_3_vl',
        'model_tag': 'Qwen3-VL-8B',
        'model_name': 'Qwen3-VL-8B-Instruct',
    },
}

openai.organization = config.openai_organization
media = pathlib.Path(__file__).parents[1] / "third_party"


def clean_text(text):
    text = text.replace('-', ' ')
    # Remove non-letter and non-space characters using regex, then convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return cleaned_text


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type


def parse_sectioned_prompt(s):
    '''
    Have sections separated by headers (lines starting with # ).
    The function parses the string into a dictionary, where each section header becomes a key,
    and the corresponding content under that header becomes the associated value.
    '''
    result = {}
    current_header = None

    for line in s.split('\n'):
        # line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


def gpt4o(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024,
          presence_penalty=0, frequency_penalty=0, logit_bias={}):
    client = OpenAI(api_key=config.openai_api_key,
                    project=config.openai_project_id)

    if img_paths != None:
        # imgs_url = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img_paths[i])}"}}
        #             for i in range(len(img_paths))]
        imgs_url = []
        for i in range(len(img_paths)):
            base64_image, mime_type = encode_image(img_paths[i])
            imgs_url.append({"type": "image_url",
                             "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
        messages = [{"role": "user",
                     "content": imgs_url + [{"type": "text", "text": prompt}], }]
    else:
        messages = [{"role": "user", "content": prompt}]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            response = client.chat.completions.create(model="gpt-4o-2024-08-06",
                                                      messages=messages,
                                                      temperature=temperature,
                                                      n=n,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      presence_penalty=presence_penalty,
                                                      frequency_penalty=frequency_penalty,
                                                      logit_bias=logit_bias
                                                      )
            num_attempts = 5
            return [response.choices[i].message.content for i in range(n)]

        except BadRequestError as be:
            print(f"BadRequestError: {be}")
            continue
        except openai.RateLimitError as e:
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"OpenAI server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)  # Wait for 5 seconds before the next attempt
            continue


def google_gemini(prompt, img_paths=None, few_shot_files=None, temperature=0.7, n=1, top_p=1, max_tokens=1024):
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    from google.generativeai.types.generation_types import StopCandidateException
    from google.generativeai import protos

    genai.configure(api_key=config.gemini_api_key)

    safety_settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": 'BLOCK_NONE'},
                       {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": 'BLOCK_NONE'}]

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

    if img_paths != None:
        # imgs = [genai.upload_file(media / image) for image in img_paths]
        imgs = [PIL.Image.open(image) for image in img_paths]
        if few_shot_files != None:
            messages = few_shot_files + imgs + [prompt]
        else:
            messages = imgs + [prompt]
    else:
        messages = prompt

    num_attempts = 0
    while num_attempts < 10:
        num_attempts += 1
        try:
            response = model.generate_content(messages,
                                              generation_config=genai.GenerationConfig(temperature=temperature,
                                                                                       top_p=top_p,
                                                                                       max_output_tokens=max_tokens),
                                              safety_settings=safety_settings
                                              )
            FinishReason = protos.Candidate.FinishReason
            if response.candidates:
                if (response.candidates[0].finish_reason == FinishReason.STOP
                        or response.candidates[0].finish_reason == FinishReason.MAX_TOKENS):
                    out = response.text
                    num_attempts = 10
                    return [out]
                else:
                    if not response.candidates:
                        print("Generate issue: No candidates returned in response.")
                    else:
                        print(f"Generate issue {response.candidates[0].finish_reason}")
                    time.sleep(1)

        except StopCandidateException as e:
            if e.args[0].finish_reason == 3:  # Block reason is safety
                print('Blocked for Safety Reasons')
                time.sleep(1)
        except ResourceExhausted as e:  # Too many requests, wait for a minute
            print("Resource Exhausted, wait for a minute to continue...")
            time.sleep(60)
        except Exception as e:
            print(f"Other issue: {e}")
            time.sleep(1)
    return [None]


def clear_gemini_img_files(verbose=False):
    import google.generativeai as genai

    genai.configure(api_key=config.gemini_api_key)
    for f in genai.list_files():
        myfile = genai.get_file(f.name)
        myfile.delete()
        if verbose:
            print("Deleted", f.name)


def get_gemini_upload_file(img_paths):
    import google.generativeai as genai

    genai.configure(api_key=config.gemini_api_key)

    files, file_names = [], []
    for image in img_paths:
        file = genai.upload_file(media / image)
        file_name = file.name
        files.append(file)
        file_names.append(file_name)

    return files, file_names


def _get_mllm_bot(model_name=None):
    """
    获取或创建 MLLMBot 单例实例
    
    Args:
        model_name: 模型名称，支持:
            - 'Qwen2.5-VL-7B-Instruct' (默认)
            - 'Qwen3-VL-8B-Instruct'
    """
    global _mllm_bot_instance, _current_model_name
    
    # 默认使用 Qwen2.5-VL
    if model_name is None:
        model_name = os.environ.get('AUTOSEP_MODEL', 'Qwen2.5-VL-7B-Instruct')
    
    # 如果模型已加载且名称匹配，直接返回
    if _mllm_bot_instance is not None and _current_model_name == model_name:
        return _mllm_bot_instance
    
    # 检查模型是否支持
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"支持的模型: {list(SUPPORTED_MODELS.keys())}"
        )
    
    # 加载新模型
    model_config = SUPPORTED_MODELS[model_name]
    print(f"[api_utils] 加载模型: {model_name}")
    
    # 动态导入模块
    import importlib
    module = importlib.import_module(model_config['module'])
    MLLMBot = module.MLLMBot
    
    _mllm_bot_instance = MLLMBot(
        model_tag=model_config['model_tag'],
        model_name=model_config['model_name'],
        pai_enable_attn=False,
        device='cuda',
        device_id=0,
        bit8=False,
        max_answer_tokens=1024
    )
    _current_model_name = model_name
    
    return _mllm_bot_instance


def sglang_model(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024, model_name='sglang_qwen'):
    """
    使用本地 Qwen VL 模型进行推理
    
    Args:
        prompt: 输入提示
        img_paths: 图片路径列表
        temperature: 温度参数（当前实现不支持，保留接口兼容）
        n: 生成数量（当前实现固定为1）
        top_p: top_p参数（当前实现不支持，保留接口兼容）
        max_tokens: 最大生成token数
        model_name: 模型名称（保留接口兼容）
    
    Returns:
        list: 包含生成文本的列表
    """
    if 'qwen' not in model_name.lower():
        raise Exception(f'Unsupported model: {model_name}')
    
    # 从环境变量获取模型名称
    actual_model = os.environ.get('AUTOSEP_MODEL', 'Qwen2.5-VL-7B-Instruct')
    mllm_bot = _get_mllm_bot(actual_model)
    
    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            if img_paths is None:
                # 纯文本推理
                response = mllm_bot.call_llm(prompt)
                return [response.strip()]
            else:
                # 图文推理
                raw_image = PIL.Image.open(img_paths[0])
                reply, _ = mllm_bot.describe_attribute(raw_image, prompt, max_new_tokens=max_tokens)
                if isinstance(reply, list):
                    return [reply[0].strip()]
                return [reply.strip()]

        except Exception as e:
            print(f"MLLMBot inference error: {e}")
            if num_attempts < 5:
                time.sleep(5)
            continue
    
    return [None]


def sglang_model1(prompt, img_paths=None, temperature=0.7, n=1, top_p=1, max_tokens=1024, model_name='sglang_qwen'):
    if 'qwen' in model_name:
        client = OpenAI(base_url=f"http://localhost:30000/v1", api_key="None")
    else:
        raise Exception(f'Unsupported task: {model_name}')

    if img_paths != None:
        imgs_url = []
        for i in range(len(img_paths)):
            imgs_url.append({"type": "image_url", "image_url": {"url": img_paths[i]}})
        messages = [{"role": "user",
                     "content": imgs_url + [{"type": "text", "text": prompt}], }]
    else:
        messages = [{"role": "user", "content": prompt}]

    num_attempts = 0
    while num_attempts < 5:
        num_attempts += 1
        try:
            response = client.chat.completions.create(model="Qwen/Qwen2-VL-72B-Instruct",
                                                      messages=messages,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      )
            num_attempts = 5
            return [response.choices[0].message.content]

        except Exception as e:
            print(f"SGLang server offers this error: {e}")
            if num_attempts < 5:
                time.sleep(5)
            continue
