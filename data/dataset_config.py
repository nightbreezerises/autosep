"""
数据集配置映射
将简短的数据集名称映射到实际的目录名和配置
"""

# 数据集名称映射: 简称 -> 实际目录名
DATASET_MAPPING = {
    # 细粒度分类
    'cub': 'CUB_200_2011',
    'dog': 'dogs_120',
    'flower': 'flowers_102',
    'car': 'car_196',
    'pet': 'pet_37',
    'aircraft': 'fgvc_aircraft',
    'food': 'food_101',
    
    # 通用分类
    'caltech101': 'caltech101',
    'caltech256': 'caltech256',
    'dtd': 'dtd',
    'eurosat': 'eurosat',
    'ucf': 'ucf101',
    'sun397': 'SUN397',
    'birdsnap': 'birdsnap',
    
    # ImageNet 系列
    'imagenet_1k': 'ImageNet_1k',
    'imagenet_a': 'ImageNet_A',
    'imagenet_r': 'ImageNet_R',
    'imagenet_sketch': 'ImageNet_Sketch',
    'imagenet_v2': 'ImageNet_v2',
}

# 数据集信息
DATASET_INFO = {
    'cub': {'name': 'CUB-200-2011', 'classes': 200, 'type': '鸟类细粒度分类'},
    'dog': {'name': 'Stanford Dogs', 'classes': 120, 'type': '狗品种分类'},
    'flower': {'name': 'Oxford Flowers', 'classes': 102, 'type': '花卉分类'},
    'car': {'name': 'Stanford Cars', 'classes': 196, 'type': '汽车型号分类'},
    'pet': {'name': 'Oxford Pets', 'classes': 37, 'type': '宠物分类'},
    'aircraft': {'name': 'FGVC Aircraft', 'classes': 100, 'type': '飞机型号分类'},
    'food': {'name': 'Food-101', 'classes': 101, 'type': '食物分类'},
    'caltech101': {'name': 'Caltech-101', 'classes': 101, 'type': '通用物体分类'},
    'caltech256': {'name': 'Caltech-256', 'classes': 256, 'type': '通用物体分类'},
    'dtd': {'name': 'DTD', 'classes': 47, 'type': '纹理分类'},
    'eurosat': {'name': 'EuroSAT', 'classes': 10, 'type': '卫星图像分类'},
    'ucf': {'name': 'UCF-101', 'classes': 101, 'type': '动作识别'},
    'sun397': {'name': 'SUN397', 'classes': 397, 'type': '场景分类'},
    'birdsnap': {'name': 'Birdsnap', 'classes': 500, 'type': '鸟类分类'},
    'imagenet_1k': {'name': 'ImageNet-1K', 'classes': 1000, 'type': '通用分类'},
    'imagenet_a': {'name': 'ImageNet-A', 'classes': 200, 'type': '对抗样本'},
    'imagenet_r': {'name': 'ImageNet-R', 'classes': 200, 'type': '渲染图像'},
    'imagenet_sketch': {'name': 'ImageNet-Sketch', 'classes': 1000, 'type': '素描图像'},
    'imagenet_v2': {'name': 'ImageNet-V2', 'classes': 1000, 'type': 'ImageNet变体'},
}


def get_dataset_dir(dataset_name):
    """
    获取数据集的实际目录名
    
    Args:
        dataset_name: 数据集简称 (如 'cub', 'dog')
    
    Returns:
        实际目录名 (如 'CUB_200_2011', 'dogs_120')
    """
    name = dataset_name.lower()
    if name in DATASET_MAPPING:
        return DATASET_MAPPING[name]
    # 如果不在映射中，直接返回原名
    return dataset_name


def get_dataset_info(dataset_name):
    """获取数据集信息"""
    name = dataset_name.lower()
    return DATASET_INFO.get(name, {'name': dataset_name, 'classes': 'unknown', 'type': 'unknown'})


def list_supported_datasets():
    """列出所有支持的数据集"""
    return list(DATASET_MAPPING.keys())
