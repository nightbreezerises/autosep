import os
import sys
import warnings
from google import genai
from google.genai.errors import APIError
from typing import List, Dict, Any, Union
from PIL import Image

# ----------------------------------------------------------------------
# ⚠️ 注意: Gemini 是一个云端 API 服务，与 Qwen 的本地 MLLM 推理模式完全不同。
# 以下代码已移除所有本地模型加载、GPU/CPU、PyTorch/Transformers相关的逻辑。
# 仅保留 API 客户端初始化和调用逻辑，并遵循您提供的 MLLMBot 结构。
# ----------------------------------------------------------------------

proxy_host = "127.0.0.1"
proxy_port = 27376  # 代理端口，方便用户更改
# 等价于: export http_proxy="http://127.0.0.1:PORT" https_proxy="http://127.0.0.1:PORT"
# 仅需修改 proxy_port 即可切换端口，默认代理控制台: 10.82.1.223:19136/ui
print(f"[Gemini Debug] 当前默认代理: http://{proxy_host}:{proxy_port}")
print("[Gemini Debug] 如需修改，请在文件顶部调整 proxy_port 或 proxy_host 后重新运行。")

# 抑制 warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 映射 Gemini 模型名称
GEMINI_MODELS = {
    'gemini-2.5-pro': 'gemini-2.5-pro', # 强大的模型
    'gemini-2.5-flash': 'gemini-2.5-flash', # 高速/低延迟模型
    # 也可以添加其他版本，例如：
    # 'gemini-1.5-pro': 'gemini-1.5-pro',
}

api_key = "AIzaSyAjtpXhIfF_y-RvTNFTDNocOTB7hhQ4l6s"

SYSTEM_INSTRUCTION = "You are a helpful assistant."

# Gemini API 的重试逻辑可以集成在调用函数内部或外部，这里使用一个简单版本。
# Google GenAI SDK 本身通常会处理网络级的重试，但我们可以在应用层添加逻辑。
# 由于去除了 tenacity 库依赖，这里使用简单的 try/except 循环实现重试。
MAX_RETRIES = 3


def _setup_proxy_env():
    def _clear_socks_env():
        removed = False
        for key in ("all_proxy", "ALL_PROXY", "socks_proxy", "SOCKS_PROXY"):
            val = os.environ.pop(key, None)
            if val:
                removed = True
                print(f"[Gemini Debug] 已移除 {key}={val} (避免触发 SOCKS 代理)")
        return removed

    if not proxy_port:
        _clear_socks_env()
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            if key in os.environ:
                os.environ.pop(key)
                print(f"[Gemini Debug] 已清理 {key}，使用系统默认网络")
        print("[Gemini Debug] 未设置代理端口，直接连接 Gemini API")
        return

    proxy_url = f"http://{proxy_host}:{proxy_port}"
    _clear_socks_env()

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        current = os.environ.get(key)
        if current != proxy_url:
            os.environ[key] = proxy_url

    print("[Gemini Debug] 已配置代理环境变量:")
    print(f"  - export http_proxy=\"{proxy_url}\"")
    print(f"  - export https_proxy=\"{proxy_url}\"")


def prepare_gemini_message(main_prompt: str) -> List[Dict[str, str]]:
    """
    为 Gemini API 准备消息格式。
    Gemini 聊天 API 使用 List[Content] 结构，每个 Content 包含 role 和 parts。
    这里简化为单轮的 system/user 结构，类似于 Qwen 的 chat log 构造。
    """
    messages = [
        {
            "role": "user",
            "parts": [
                {"text": f"SYSTEM_INSTRUCTION: {SYSTEM_INSTRUCTION}\n{main_prompt}"}
            ]
        }
    ]
    # 官方推荐的系统指令传递方式是在 Client 或 Config 中，这里为了仿照旧代码结构，
    # 暂时将系统指令放入用户提示中，或者在 __call_llm 中使用 system_instruction 参数。
    return main_prompt # 返回原始 prompt，在调用时处理结构


def trim_answer(answer: Union[str, List[str]]) -> str:
    """
    清理和修剪模型的回复。
    """
    if isinstance(answer, list):
        # 兼容 __call_llm 返回列表的情况
        answer = answer[0]
        
    # 移除可能的分隔符或多余内容
    answer = answer.split('Question:')[0].replace('\n', ' ').strip()
    return answer


class MLLMBot:
    # 仿照 Qwen 的 MLLMBot 结构，但专注于 API 调用
    def __init__(self, model_tag: str, model_name: str, device: str = 'api', max_answer_tokens: int = 256):
        
        self.model_tag = model_tag # 如 'gemini-2.5-pro'
        self.model_name = model_name # 如 'Gemini 2.5 Pro (API)'
        self.max_answer_tokens = max_answer_tokens # 对应 max_output_tokens
        self.device = device
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

        if self.model_tag not in GEMINI_MODELS:
            raise ValueError(f"Model tag '{model_tag}' not supported. Available: {list(GEMINI_MODELS.keys())}")
        
        # 确保 API Key 已设置
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        _setup_proxy_env()

        # 初始化 Gemini 客户端
        self.client = genai.Client(api_key=api_key)
        self.api_model_name = GEMINI_MODELS[self.model_tag]
        
        # 记录配置信息
        print("\n================= 模型初始化（MLLMBot - Gemini API） =================")
        print(f"📌 模型标识（model_tag）: {model_tag}")
        print(f"📌 模型名称（model_name）: {model_name}")
        print(f"🖥️ 设备: {self.device} (云端 API)")
        print(f"🔧 最大生成长度 max_answer_tokens: {self.max_answer_tokens}")
        print("🚀 客户端加载完成！")
        print("========================================================\n")
        
        # API 服务没有 GPU/CPU 清理，但保留方法签名以仿照 Qwen 风格
        
    def __del__(self):
        """析构函数（API模式下无实际操作）"""
        pass
    
    def cleanup(self):
        """手动清理内存（API模式下无实际操作）"""
        pass
        
    def get_name(self):
        return self.model_name
    
    def _log_image_debug(self, images: List[Image.Image]):
        if not images:
            print("[Gemini Debug] 无图像输入，按纯文本模式推理")
            return
        details = []
        for idx, img in enumerate(images):
            if isinstance(img, Image.Image):
                width, height = img.size
                details.append(f"#{idx + 1}:{width}x{height},mode={img.mode}")
            else:
                details.append(f"#{idx + 1}:非PIL对象({type(img)})")
        print(f"[Gemini Debug] 接收到 {len(images)} 张图像 -> {' | '.join(details)}")

    def _log_prompt_debug(self, prompt: str, max_new_tokens: int):
        prompt_clean = ' '.join(prompt.strip().split())
        preview = (prompt_clean[:200] + '...') if len(prompt_clean) > 200 else prompt_clean
        print("[Gemini Debug] 文本提示信息：")
        print(f"  - 字符数: {len(prompt)}")
        print(f"  - Max New Tokens: {max_new_tokens}")
        print(f"  - Preview: {preview}")

    def _log_api_response(self, response, total_tokens: int):
        usage = getattr(response, 'usage_metadata', None)
        prompt_tokens = getattr(usage, 'prompt_token_count', 0) if usage else 0
        # candidates_token_count 可能为 None，需要处理
        completion_tokens = getattr(usage, 'candidates_token_count', None) if usage else 0
        if completion_tokens is None:
            completion_tokens = 0
        # Gemini 2.5 Flash 会使用思考 token
        thoughts_tokens = getattr(usage, 'thoughts_token_count', 0) if usage else 0
        if thoughts_tokens is None:
            thoughts_tokens = 0
        candidates = getattr(response, 'candidates', [])
        finish_reason = candidates[0].finish_reason if candidates else 'unknown'
        response_text = response.text if response.text else ""
        reply_preview = response_text.strip().replace('\n', ' ')
        if len(reply_preview) > 200:
            reply_preview = reply_preview[:200] + '...'
        print("[Gemini Debug] API调用成功：")
        print(f"  - 使用模型: {self.api_model_name}")
        print(f"  - Tokens Used (total/prompt/output/thoughts): {total_tokens}/{prompt_tokens}/{completion_tokens}/{thoughts_tokens}")
        print(f"  - Finish Reason: {finish_reason}")
        print(f"  - Response Preview: {reply_preview}")
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        self._log_usage_summary()

    def _log_usage_summary(self):
        print("[Gemini Debug] 累计用量统计：")
        print(f"  - 总请求次数: {self.total_requests}")
        print(f"  - 累计 Prompt Tokens: {self.total_prompt_tokens}")
        print(f"  - 累计 Output Tokens: {self.total_output_tokens}")

    def _log_quota_warning(self, error: Exception):
        error_str = str(error)
        lower_err = error_str.lower()
        quota_keywords = ["insufficient_quota", "quota", "rate limit", "exceeded"]
        if any(keyword in lower_err for keyword in quota_keywords):
            print("⚠️ [Gemini Debug] 可能触发配额/速率限制，请检查 Google AI Studio 中的使用额度。")
            print("  - 建议: 减少 batch 大小、降低 max_output_tokens，或升级账号额度。")

    # 仿照 Qwen 的 __call_qwen2_5 方法
    def __call_llm(self, raw_image: Union[Image.Image, None], prompt: str, max_new_tokens: int = 256) -> List[str]:
        
        contents = []
        image_payload: List[Image.Image] = []
        
        # 1. 处理图像 (多模态输入)
        if raw_image:
            # 兼容单图和多图，这里假设 raw_image 是 PIL.Image 或 PIL.Image 列表
            if not isinstance(raw_image, list):
                raw_image = [raw_image]

            for img in raw_image:
                # Gemini API 直接接受 PIL Image 对象作为 parts
                image_payload.append(img)

        self._log_image_debug(image_payload)

        contents.extend(image_payload)
        # 2. 处理文本 Prompt
        contents.append(prompt)
        self._log_prompt_debug(prompt, max_new_tokens)
        
        # 3. 配置生成参数 (对应 Qwen 的 generate 参数)
        # 注意: Gemini 2.5 Flash 使用思考 token，需要增加 max_output_tokens
        # 思考 token 可能占用 1000+ tokens，所以需要大幅预留空间
        # 例如：max_new_tokens=256 时，思考可能用 1000+，实际输出需要 256，总共需要 1500+
        effective_max_tokens = max(max_new_tokens + 2048, 2560)  # 预留 2048 给思考，至少 2560
        config = {
            "max_output_tokens": effective_max_tokens,
            "temperature": 0.9, # 默认值，如果需要可作为参数传入
            "system_instruction": SYSTEM_INSTRUCTION # 推荐的系统指令传递方式
        }
        
        # 4. 执行 API 调用 (带重试逻辑)
        reply = [""]
        total_tokens = 0
        print(f"[Gemini Debug] 即将调用云端模型 {self.api_model_name}，最多重试 {MAX_RETRIES} 次")
        
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[Gemini Debug] 第 {attempt + 1} 次尝试，发送请求...")
                response = self.client.models.generate_content(
                    model=self.api_model_name,
                    contents=contents,
                    config=config
                )
                
                # 提取回复和Token数
                response_text = response.text
                if response_text is None:
                    # 检查是否有候选内容
                    candidates = getattr(response, 'candidates', [])
                    if candidates and hasattr(candidates[0], 'content'):
                        parts = getattr(candidates[0].content, 'parts', [])
                        if parts and hasattr(parts[0], 'text'):
                            response_text = parts[0].text
                    if response_text is None:
                        print(f"[Gemini Debug] 警告: API 返回空内容，可能被安全过滤")
                        print(f"[Gemini Debug] 响应对象: {response}")
                        response_text = ""
                
                reply = [response_text]
                total_tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
                
                # 打印 Token 统计 (仿照 Qwen 打印内存/Token)
                self._log_api_response(response, total_tokens)
                break # 成功，退出重试循环
                
            except APIError as e:
                print(f"API Error (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                self._log_quota_warning(e)
                if attempt < MAX_RETRIES - 1:
                    # 仅在非最后一次尝试时等待
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise
            except Exception as e:
                print(f"Unexpected Error: {e}")
                raise

        # API 模式下无法获取 total_tokens 的增量，这里返回一个元组，仿照复杂调用的习惯。
        return reply, total_tokens
    

    # 以下方法仿照 Qwen MLLMBot 的外部接口，调用 __call_llm

    def answer_chat_log(self, raw_image, chat_log: Dict[str, List[str]], n_context: int = -1):
        """
        处理聊天历史记录，并生成回复。
        """
        # ⚠️ 注意: Gemini API 的多轮聊天最好使用 client.chats.create()。
        # 这里为了仿照 Qwen 的 get_chat_log 结构，我们将其打包成一个长文本 prompt。
        
        # 仿照 Qwen 的 get_chat_log 构造文本
        history_str = self._format_chat_log(chat_log, n_context)
        
        gemini_prompt = '\n'.join([
            history_str, 
            "Please provide a concise answer to the last question."
        ])

        reply_list, _ = self.__call_llm(raw_image, gemini_prompt, max_new_tokens=self.max_answer_tokens)
        
        reply = reply_list[0]
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def describe_attribute(self, raw_image: Union[Image.Image, List[Image.Image]], attr_prompt: str, max_new_tokens: int = 256):
        """
        描述图像属性或回答问题。
        """
        reply_list, _ = self.__call_llm(raw_image, attr_prompt, max_new_tokens)
        trimmed_reply = trim_answer(reply_list)
        return reply_list[0], trimmed_reply

    def _format_chat_log(self, chat_log: Dict[str, List[str]], last_n: int = -1) -> str:
        """
        仿照 Qwen 的 get_chat_log 逻辑，将聊天历史转换为文本。
        """
        questions = chat_log.get('questions', [])
        answers = chat_log.get('answers', [])
        
        n_addition_q = len(questions) - len(answers)
        
        # 截断逻辑
        if last_n > 0:
            answers = answers[-last_n:]
            questions = questions[-(last_n + n_addition_q):]
        elif last_n == 0:
            answers = []
            questions = questions[-1:] if n_addition_q else []

        template = 'User: {} \nAssistant: {} \n'
        chat_log_str = ''

        for i in range(len(answers)):
            chat_log_str += template.format(questions[i], answers[i])
            
        if n_addition_q:
            chat_log_str += 'User: {}'.format(questions[-1])
        else:
            # 移除末尾的换行和空格
            chat_log_str = chat_log_str.strip() 

        return chat_log_str


# ----------------------------------------------------------------------
# 示例用法 (仿照 test_get_llm_output)
# ----------------------------------------------------------------------
def test_mllm_output():
    
    # ⚠️ 请确保您已设置 GEMINI_API_KEY 环境变量，且 PIL 库已安装
    # pip install Pillow
    
    try:
        from PIL import Image
        
        # 准备一个虚拟图像对象 (Gemini API支持多模态)
        # 实际使用中，你需要从文件加载真实的图片
        try:
            raw_image = Image.new('RGB', (100, 100), color = 'red')
        except:
            raw_image = None
            print("Warning: PIL Image creation failed. Running text-only test.")


        # 初始化 MLLMBot
        model_tag = "gemini-2.5-pro"
        model_name = "Gemini 2.5 Pro (API Test)"
        bot_llm = MLLMBot(model_tag=model_tag, model_name=model_name, max_answer_tokens=1024)
        
        prompt = "描述这张图片的内容，或者如果图片为空，请回答 '图片内容为空'。"
        
        print("\n--- 发起 API 调用 ---")
        reply, trimmed_reply = bot_llm.describe_attribute(raw_image, prompt, max_new_tokens=512)
        
        print("\n--- 结果 ---")
        print(f"Model: {bot_llm.get_name()}")
        print(f"Prompt: {prompt}")
        print(f"Reply: {reply}")
        print(f"Trimmed Reply: {trimmed_reply}")
        
    except EnvironmentError as e:
        print(f"\n[错误]: {e}")
        print("请先设置 GEMINI_API_KEY 环境变量。")
    except APIError as e:
        print(f"\n[API 错误]: {e}")
        print("请检查您的 API Key 是否有效，以及模型名称是否正确 (gemini-2.5-pro)。")
    except Exception as e:
        print(f"\n[运行时错误]: {e}")
        
if __name__ == '__main__':
    test_mllm_output()