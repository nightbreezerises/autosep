"""
通用数据集加载器
支持从 split_dataset_images.json 加载任意数据集
"""
import os
import json
import random
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import f1_score

from .dataset_config import get_dataset_dir

random.seed(42)


class GenericDatasetLoader:
    """
    通用数据集加载器
    
    数据集目录结构:
    {data_dir}/
    └── {dataset_dir}/
        └── {dataset_dir}/
            ├── images/
            └── images_split/
                └── split_dataset_images.json
    
    split_dataset_images.json 格式:
    {
        "train": [["类别名", 类别ID, "相对图片路径"], ...],
        "val": [...],
        "test": [...]
    }
    """
    
    def __init__(self, data_dir, dataset_name, max_threads=8):
        """
        Args:
            data_dir: 数据集根目录 (如 ./datasets)
            dataset_name: 数据集名称 (如 'cub' 或 'CUB_200_2011')
            max_threads: 最大线程数
        """
        self.data_dir = os.path.abspath(data_dir)
        self.dataset_name = dataset_name
        self.max_threads = max_threads
        
        # 获取实际目录名
        self.dataset_dir = get_dataset_dir(dataset_name)
        
        # 数据集路径：CUB 是双层嵌套，其他数据集是单层
        # 双层: datasets/CUB_200_2011/CUB_200_2011/images_split/
        # 单层: datasets/dogs_120/images_split/
        double_nested_path = os.path.join(self.data_dir, self.dataset_dir, self.dataset_dir)
        single_nested_path = os.path.join(self.data_dir, self.dataset_dir)
        
        # 划分文件名：优先使用 split_datasets_images.json，兼容 split_dataset_images.json
        split_file_names = ['split_datasets_images.json', 'split_dataset_images.json']
        
        # 查找划分文件
        self.dataset_path = None
        self.split_file = None
        
        for path in [single_nested_path, double_nested_path]:
            for split_name in split_file_names:
                candidate = os.path.join(path, 'images_split', split_name)
                if os.path.exists(candidate):
                    self.dataset_path = path
                    self.split_file = candidate
                    break
            if self.split_file:
                break
        
        if not self.split_file:
            raise FileNotFoundError(
                f"划分文件不存在，已尝试:\n"
                f"  - {os.path.join(single_nested_path, 'images_split', 'split_datasets_images.json')}\n"
                f"  - {os.path.join(double_nested_path, 'images_split', 'split_datasets_images.json')}\n"
            )
        
        with open(self.split_file, 'r') as f:
            self.split_data = json.load(f)
        
        # 构建类别映射
        self._build_class_mapping()
        
        print(f"[DatasetLoader] 加载数据集: {dataset_name} -> {self.dataset_dir}")
        print(f"  - 路径: {self.dataset_path}")
        print(f"  - 划分文件: {os.path.basename(self.split_file)}")
        print(f"  - 类别数: {self.num_classes}")
        print(f"  - 训练集: {len(self.split_data.get('train', []))} 样本")
        print(f"  - 验证集: {len(self.split_data.get('val', []))} 样本")
        print(f"  - 测试集: {len(self.split_data.get('test', []))} 样本")
    
    def _build_class_mapping(self):
        """构建类别名称和ID的映射"""
        self.class_names = {}  # id -> name
        self.name_to_id = {}   # name -> id
        
        for split in ['train', 'val', 'test']:
            if split not in self.split_data:
                continue
            for item in self.split_data[split]:
                class_name, class_id, _ = item
                if class_id not in self.class_names:
                    self.class_names[class_id] = class_name
                    self.name_to_id[class_name] = class_id
        
        self.num_classes = len(self.class_names)
    
    def stringify_prediction(self, pred):
        """将预测ID转换为类别名称"""
        return self.class_names.get(pred, f"Unknown_{pred}")
    
    def get_examples(self, mode='train'):
        """获取指定划分的所有样本"""
        if mode not in self.split_data:
            raise ValueError(f"不支持的划分: {mode}, 可用: {list(self.split_data.keys())}")
        
        exs = []
        for i, item in enumerate(self.split_data[mode]):
            class_name, class_id, rel_img_path = item
            img_path = os.path.join(self.dataset_path, rel_img_path)
            exs.append({
                'id': f'{mode}-{i}',
                'label': class_id,
                'label_name': class_name,
                'img_path': img_path
            })
        return exs
    
    def get_even_exs(self, mode='train', n_exs=10):
        """获取每个类别均匀采样的样本"""
        if mode not in self.split_data:
            raise ValueError(f"不支持的划分: {mode}")
        
        # 按类别分组
        class_samples = {}
        for i, item in enumerate(self.split_data[mode]):
            class_name, class_id, rel_img_path = item
            if class_id not in class_samples:
                class_samples[class_id] = []
            class_samples[class_id].append((i, class_name, rel_img_path))
        
        # 每个类别取 n_exs 个样本
        exs = []
        for class_id in sorted(class_samples.keys()):
            samples = class_samples[class_id][:n_exs]
            for idx, class_name, rel_img_path in samples:
                img_path = os.path.join(self.dataset_path, rel_img_path)
                exs.append({
                    'id': f'{mode}-{idx}',
                    'label': class_id,
                    'label_name': class_name,
                    'img_path': img_path
                })
        
        return exs
    
    def get_few_shot_examples(self, n_shots=1, seed=42):
        """获取 few-shot 样本"""
        random.seed(seed)
        
        class_samples = {}
        for i, item in enumerate(self.split_data.get('train', [])):
            class_name, class_id, rel_img_path = item
            if class_id not in class_samples:
                class_samples[class_id] = []
            class_samples[class_id].append((i, class_name, rel_img_path))
        
        exs = [[] for _ in range(self.num_classes)]
        for class_id in sorted(class_samples.keys()):
            samples = class_samples[class_id]
            selected = random.sample(samples, min(n_shots, len(samples)))
            for idx, class_name, rel_img_path in selected:
                img_path = os.path.join(self.dataset_path, rel_img_path)
                exs[class_id].append({
                    'id': f'train-{idx}',
                    'label': class_id,
                    'label_name': class_name,
                    'img_path': img_path
                })
        
        return exs
    
    def get_attr(self, mode, prompt, exs, gpt_generator=None, generate=False, exp=1):
        """获取或生成属性描述"""
        import generator
        
        if generate:
            attrs = {}
            attribute_cache = {f'{prompt}': {}}
            attribute_cache = generator.parallel_generate(
                gpt_generator, prompt, exs, attribute_cache, self.max_threads
            )
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            cache_file = os.path.join(
                os.path.dirname(self.data_dir), 'autosep', 'results',
                f'{exp}_{self.dataset_name}', f'{exp}_{mode}_attr.json'
            )
            with open(cache_file, 'r') as f:
                attr = json.load(f)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs
    
    # ========== 评估方法 ==========
    
    def run_evaluate(self, predictor, prompt, exs, pred_prompts=None, 
                     attribute_cache=None, model_name='sglang_qwen'):
        """
        运行评估
        
        注意: 本地模型 (sglang_qwen) 使用顺序执行，因为:
        1. GPU 模型是单例，无法多进程共享
        2. 局部函数无法被 pickle 序列化
        """
        labels, preds, texts, attributes = [], [], [], []
        
        # 本地模型使用顺序执行（GPU 模型无法多进程）
        for ex in tqdm(exs, desc='评估中'):
            img_path = ex['img_path']
            
            if attribute_cache is not None:
                pred_prompt = pred_prompts[f'{prompt}']
                attr = attribute_cache[f'{prompt}'][f'{ex}']
            else:
                pred_prompt = prompt
                attr = None
            
            pred = predictor.inference(pred_prompt, [img_path], attr)
            
            if pred is not None:
                texts.append(ex['img_path'])
                labels.append(ex['label'])
                preds.append(pred)
                if attribute_cache is not None:
                    attributes.append(attr)
        
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds, attributes
    
    def evaluate(self, predictor, prompt, test_exs, pred_prompts=None, 
                 attribute_cache=None, model_name='sglang_qwen'):
        """评估（带重试）"""
        import requests
        while True:
            try:
                return self.run_evaluate(
                    predictor, prompt, test_exs, pred_prompts,
                    attribute_cache, model_name
                )
            except (concurrent.futures.process.BrokenProcessPool, 
                    requests.exceptions.SSLError):
                pass
    
    def compare_evaluate(self, prompt, exs, attribute_cache, model_name='sglang_qwen'):
        """
        对比评估方法
        用于 ProTeGi 优化器的 expand_candidates 阶段
        """
        from autosep.llm_text_compare import predict_with_compare, select_k_from_n_excluding_i
        
        false_exs_dict = {}
        
        for i in range(len(exs)):
            false_idx = select_k_from_n_excluding_i(len(exs), 2, i)  # compare
            false_exs_dict[f'{exs[i]}'] = [exs[idx] for idx in false_idx]
        
        true_exs, false_exs, preds, true_attrs, false_attrs = [], [], [], [], []
        
        # 本地模型使用顺序执行（GPU 模型无法多进程）
        if 'sglang' in model_name:
            for true_ex in tqdm(exs, desc='running comparison on examples (single)'):
                for false_ex in false_exs_dict[f'{true_ex}'][:3]:
                    answer, true_ex, false_ex, prompt = predict_with_compare(
                        true_ex, false_ex, prompt,
                        attribute_cache[f'{prompt}'], model_name
                    )
                    true_exs.append(true_ex)
                    false_exs.append(false_ex)
                    preds.append(answer)
                    true_attrs.append(attribute_cache[f'{prompt}'][f'{true_ex}'])
                    false_attrs.append(attribute_cache[f'{prompt}'][f'{false_ex}'])
        else:
            # 其他模型使用并行执行
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [executor.submit(predict_with_compare, true_ex, false_ex, prompt,
                                           attribute_cache[f'{prompt}'], model_name)
                           for true_ex in exs for false_ex in false_exs_dict[f'{true_ex}']]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)),
                                      total=len(futures), desc='running comparison on examples (parallel)'):
                    answer, true_ex, false_ex, prompt = future.result()
                    true_exs.append(true_ex)
                    false_exs.append(false_ex)
                    preds.append(answer)
                    true_attrs.append(attribute_cache[f'{prompt}'][f'{true_ex}'])
                    false_attrs.append(attribute_cache[f'{prompt}'][f'{false_ex}'])
        
        return true_exs, false_exs, preds, true_attrs, false_attrs
