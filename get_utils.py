import os
import tasks
import predictors
from data import GenericDatasetLoader, get_dataset_dir, DATASET_MAPPING

# 原有的特定任务（3分类子任务）
LEGACY_TASKS = ['iNat_butterfly', 'iNat_lupine', 'CUB_cuckoo', 'CUB_vireo', 'CUB_oriole',
                'Stanford_terrier', 'vegfru_greens', 'vegfru_allium']


def is_generic_dataset(task_name):
    """判断是否为通用数据集（使用简称或完整目录名）"""
    # 检查是否在映射中
    if task_name.lower() in DATASET_MAPPING:
        return True
    # 检查是否是映射的值（完整目录名）
    if task_name in DATASET_MAPPING.values():
        return True
    return False


def get_task_class(args):
    """
    获取任务类
    
    支持两种模式:
    1. 通用数据集: 使用 GenericDatasetLoader 从 split_dataset_images.json 加载
    2. 原有特定任务: 使用原有的 tasks.py 中的类
    """
    task_name = args.task_name
    
    # 检查是否为通用数据集
    if is_generic_dataset(task_name):
        return GenericDatasetLoader(
            data_dir=args.data_dir,
            dataset_name=task_name,
            max_threads=getattr(args, 'max_threads', 8)
        )
    
    # 原有的特定任务
    if 'iNat' in task_name:
        return tasks.iNaturalistMultiTask(args.data_dir, None, args.max_threads, args)
    elif 'CUB' in task_name:
        return tasks.CUBMultiTask(args.data_dir, None, args.max_threads, args)
    elif 'Stanford' in task_name:
        return tasks.StanfordDogMultiTask(args.data_dir, None, args.max_threads, args)
    elif 'vegfru' in task_name:
        return tasks.VegFruMultiTask(args.data_dir, None, args.max_threads, args)
    else:
        # 尝试作为通用数据集加载
        dataset_dir = get_dataset_dir(task_name)
        split_file = os.path.join(
            args.data_dir, dataset_dir, dataset_dir, 
            'images_split', 'split_dataset_images.json'
        )
        if os.path.exists(split_file):
            print(f"[get_task_class] 检测到 {task_name} 的划分文件，使用通用加载器")
            return GenericDatasetLoader(
                data_dir=args.data_dir,
                dataset_name=task_name,
                max_threads=getattr(args, 'max_threads', 8)
            )
        raise Exception(f'Unsupported task: {task_name}')


def get_predictor(configs):
    """
    获取预测器
    
    通用数据集使用 MultiClassPredictor（支持多分类）
    原有任务使用 ThreeClassPredictor（3分类）
    """
    task_name = configs['task_name']
    
    # 通用数据集使用多分类预测器
    if is_generic_dataset(task_name):
        return predictors.MultiClassPredictor(configs)
    
    # 原有特定任务
    if ('iNat' in task_name or 'CUB' in task_name or 
        'Stanford' in task_name or 'vegfru' in task_name):
        return predictors.ThreeClassPredictor(configs)
    else:
        # 默认使用多分类预测器
        return predictors.MultiClassPredictor(configs)


def get_exs(args, task):
    """
    获取训练/验证/测试样本
    """
    task_name = args.task_name
    
    # 通用数据集
    if is_generic_dataset(task_name) or isinstance(task, GenericDatasetLoader):
        train_exs = task.get_even_exs('train', args.n_train)
        # 尝试获取验证集和测试集
        try:
            val_exs = task.get_even_exs('val', args.n_val)
        except (ValueError, KeyError):
            val_exs = None
        try:
            test_exs = task.get_even_exs('test', args.n_test)
        except (ValueError, KeyError):
            # 如果没有 test，尝试用 val
            test_exs = task.get_even_exs('val', args.n_test) if val_exs is None else val_exs
        return train_exs, val_exs, test_exs
    
    # 原有特定任务
    train_exs = task.get_even_exs('train', args.n_train)
    if 'CUB' in task_name or 'Stanford' in task_name:
        val_exs = task.get_even_exs('val', args.n_val)
        test_exs = task.get_even_exs('test', args.n_test)
        return train_exs, val_exs, test_exs
    elif 'iNat' in task_name:
        val_exs = None
        test_exs = task.get_even_exs('val', args.n_test)
        return train_exs, val_exs, test_exs
    elif 'vegfru' in task_name:
        val_exs = None
        test_exs = task.get_even_exs('test', args.n_test)
        return train_exs, val_exs, test_exs
    else:
        raise Exception(f'Unsupported task: {task_name}')
