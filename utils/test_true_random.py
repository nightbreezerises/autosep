#!/usr/bin/env python3
"""
真随机数生成器验证脚本
运行10000次true_random.py，验证：
1. 每次生成的10个随机数是否都不一样
2. 生成的随机数是否满足均匀分布
"""

import subprocess
import sys
import os
from collections import Counter, defaultdict
import statistics
from scipy import stats
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_true_random_once(use_blocking=False, use_direct_import=True, min_val=1, max_val=100, count=10):
    """
    运行一次true_random.py，获取生成的随机数
    
    Args:
        use_blocking: 是否使用阻塞式/dev/random
        use_direct_import: 是否直接导入模块（更快）还是通过subprocess运行
        min_val: 随机数最小值
        max_val: 随机数最大值
        count: 生成随机数的数量
    
    Returns:
        list: 生成的随机数列表，如果失败返回None
    """
    if use_direct_import:
        # 直接导入模块，效率更高
        try:
            from utils.true_random import TrueRandomGenerator
            
            # 创建新的生成器实例（每次重置以确保独立性）
            # 使用静默模式避免大量输出
            true_random = TrueRandomGenerator(
                use_blocking=use_blocking,
                fallback_to_pseudo=True,
                silent=True  # 静默模式，不打印初始化信息
            )
            
            if not true_random.is_available():
                return None
            
            # 生成指定数量的随机数
            random_numbers = []
            for i in range(count):
                num = true_random.random_int(min_val, max_val)
                random_numbers.append(num)
            
            true_random.close()
            return random_numbers if len(random_numbers) == count else None
            
        except Exception as e:
            print(f"⚠️  直接导入失败: {e}")
            # 回退到subprocess方式
            use_direct_import = False
    
    if not use_direct_import:
        # 通过subprocess运行（较慢但更独立）
        true_random_path = os.path.join(project_root, 'utils', 'true_random.py')
        
        cmd = [sys.executable, true_random_path]
        if use_blocking:
            cmd.append('--blocking')
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            if result.returncode != 0:
                return None
            
            # 从输出中提取随机数
            output = result.stdout
            random_numbers = []
            
            for line in output.split('\n'):
                if '随机数' in line and ':' in line:
                    try:
                        # 提取数字，格式如 "  随机数 1: 42"
                        parts = line.split(':')
                        if len(parts) == 2:
                            num = int(parts[1].strip())
                            random_numbers.append(num)
                    except ValueError:
                        continue
            
            return random_numbers if len(random_numbers) == 10 else None
            
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            return None
    
    return None


def check_uniqueness(all_results):
    """
    检查每次生成的随机数是否都不一样
    
    Args:
        all_results: 所有运行结果的列表，每个元素是10个随机数的列表
        
    Returns:
        dict: 统计信息
    """
    unique_count = 0
    duplicate_count = 0
    total_duplicates = 0
    
    for i, result in enumerate(all_results):
        if result is None:
            continue
        
        # 检查这10个随机数中是否有重复
        if len(result) == len(set(result)):
            unique_count += 1
        else:
            duplicate_count += 1
            # 统计重复的数量
            counter = Counter(result)
            duplicates = sum(count - 1 for count in counter.values() if count > 1)
            total_duplicates += duplicates
    
    return {
        'unique_runs': unique_count,
        'duplicate_runs': duplicate_count,
        'total_duplicates': total_duplicates,
        'unique_ratio': unique_count / len(all_results) if all_results else 0
    }


def check_uniform_distribution(all_numbers, min_val=1, max_val=100):
    """
    检查随机数是否满足均匀分布
    
    Args:
        all_numbers: 所有生成的随机数列表（扁平化）
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        dict: 统计信息
    """
    if not all_numbers:
        return None
    
    # 统计每个数字出现的次数
    counter = Counter(all_numbers)
    
    # 计算范围大小
    range_size = max_val - min_val + 1
    
    # 期望频率（均匀分布）
    expected_freq = len(all_numbers) / range_size
    
    # 计算卡方检验
    observed = [counter.get(i, 0) for i in range(min_val, max_val + 1)]
    expected = [expected_freq] * range_size
    
    # 卡方检验
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    # 计算实际频率与期望频率的差异
    freq_diff = {}
    for i in range(min_val, max_val + 1):
        observed_freq = counter.get(i, 0)
        diff = abs(observed_freq - expected_freq)
        freq_diff[i] = {
            'observed': observed_freq,
            'expected': expected_freq,
            'diff': diff,
            'diff_ratio': diff / expected_freq if expected_freq > 0 else 0
        }
    
    # 计算统计量
    mean_freq = statistics.mean(observed)
    std_freq = statistics.stdev(observed) if len(observed) > 1 else 0
    
    # 找出偏差最大的数字
    max_diff_num = max(range(min_val, max_val + 1), key=lambda x: freq_diff[x]['diff'])
    min_diff_num = min(range(min_val, max_val + 1), key=lambda x: freq_diff[x]['diff'])
    
    return {
        'total_numbers': len(all_numbers),
        'expected_freq': expected_freq,
        'mean_freq': mean_freq,
        'std_freq': std_freq,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'is_uniform': p_value > 0.05,  # p值>0.05认为满足均匀分布
        'max_diff_num': max_diff_num,
        'max_diff': freq_diff[max_diff_num],
        'min_diff_num': min_diff_num,
        'min_diff': freq_diff[min_diff_num],
        'frequency_distribution': dict(counter)
    }


def main():
    """主函数"""
    # 解析命令行参数
    total_runs = 10000
    min_val = 1
    max_val = 100
    count_per_run = 10
    
    if '--runs' in sys.argv:
        idx = sys.argv.index('--runs')
        if idx + 1 < len(sys.argv):
            try:
                total_runs = int(sys.argv[idx + 1])
            except ValueError:
                print("⚠️  无效的运行次数，使用默认值10000")
    elif '-n' in sys.argv:
        idx = sys.argv.index('-n')
        if idx + 1 < len(sys.argv):
            try:
                total_runs = int(sys.argv[idx + 1])
            except ValueError:
                print("⚠️  无效的运行次数，使用默认值10000")
    
    # 解析范围参数
    if '--range' in sys.argv:
        idx = sys.argv.index('--range')
        if idx + 1 < len(sys.argv):
            try:
                parts = sys.argv[idx + 1].split('-')
                if len(parts) == 2:
                    min_val = int(parts[0])
                    max_val = int(parts[1])
            except ValueError:
                pass
    
    # 解析每次生成的数量
    if '--count' in sys.argv:
        idx = sys.argv.index('--count')
        if idx + 1 < len(sys.argv):
            try:
                count_per_run = int(sys.argv[idx + 1])
            except ValueError:
                pass
    
    print("=" * 80)
    print("真随机数生成器验证测试")
    print("=" * 80)
    print(f"将生成 {total_runs * count_per_run:,} 个随机数（范围: {min_val}-{max_val}）")
    print(f"验证：1. 每次生成的随机数是否都不一样")
    print(f"     2. 生成的随机数是否满足均匀分布（卡方检验）")
    print("=" * 80)
    print()
    
    # 询问是否使用阻塞式
    use_blocking = '--blocking' in sys.argv or '-b' in sys.argv
    if use_blocking:
        print("⚠️  使用阻塞式/dev/random，可能会较慢...")
    else:
        print("使用非阻塞式/dev/urandom（推荐）")
    print()
    
    # 对于大规模测试，直接生成所有随机数，而不是分批次
    if total_runs * count_per_run >= 1000000:
        print(f"大规模测试模式：直接生成 {total_runs * count_per_run:,} 个随机数...")
        from utils.true_random import TrueRandomGenerator
        
        true_random = TrueRandomGenerator(
            use_blocking=use_blocking,
            fallback_to_pseudo=True,
            silent=True
        )
        
        if not true_random.is_available():
            print("❌ 真随机数生成器不可用，退出")
            sys.exit(1)
        
        all_numbers = []
        total_count = total_runs * count_per_run
        print("进度: ", end='', flush=True)
        
        for i in range(total_count):
            num = true_random.random_int(min_val, max_val)
            all_numbers.append(num)
            if (i + 1) % (total_count // 100) == 0:
                print(f"{(i+1)*100//total_count}% ", end='', flush=True)
        
        print(f"\n✓ 完成 {len(all_numbers):,} 个随机数生成")
        true_random.close()
        
        # 为了兼容性，创建all_results（虽然不用于唯一性检查）
        all_results = []
        for i in range(0, len(all_numbers), count_per_run):
            all_results.append(all_numbers[i:i+count_per_run])
    else:
        # 小规模测试：分批次运行
        all_results = []
        all_numbers = []
        
        print(f"开始运行 {total_runs} 次测试...")
        print("进度: ", end='', flush=True)
        
        for i in range(total_runs):
            if (i + 1) % max(1, total_runs // 100) == 0:
                print(f"{i+1}/{total_runs} ", end='', flush=True)
            
            result = run_true_random_once(
                use_blocking=use_blocking, 
                use_direct_import=True,
                min_val=min_val,
                max_val=max_val,
                count=count_per_run
            )
            if result:
                all_results.append(result)
                all_numbers.extend(result)
        
        print(f"\n✓ 完成 {len(all_results)} 次成功运行")
    
    print()
    
    # 验证1: 检查唯一性
    print("=" * 80)
    print("验证1: 检查每次生成的随机数是否都不一样")
    print("=" * 80)
    uniqueness_stats = check_uniqueness(all_results)
    print(f"总运行次数: {len(all_results)}")
    print(f"每次{count_per_run}个数都不重复的运行次数: {uniqueness_stats['unique_runs']}")
    print(f"存在重复的运行次数: {uniqueness_stats['duplicate_runs']}")
    print(f"总重复数字数: {uniqueness_stats['total_duplicates']}")
    print(f"唯一性比例: {uniqueness_stats['unique_ratio']:.2%}")
    
    if uniqueness_stats['unique_ratio'] > 0.9:
        print("✓ 唯一性验证通过（>90%的运行中10个数都不重复）")
    else:
        print("⚠️  唯一性验证未完全通过")
    print()
    
    # 验证2: 检查均匀分布
    print("=" * 80)
    print("验证2: 检查随机数是否满足均匀分布（卡方检验）")
    print("=" * 80)
    distribution_stats = check_uniform_distribution(all_numbers, min_val=min_val, max_val=max_val)
    
    if distribution_stats:
        print(f"总随机数: {distribution_stats['total_numbers']}")
        print(f"期望频率（每个数字）: {distribution_stats['expected_freq']:.2f}")
        print(f"实际平均频率: {distribution_stats['mean_freq']:.2f}")
        print(f"频率标准差: {distribution_stats['std_freq']:.2f}")
        print(f"卡方统计量: {distribution_stats['chi2_statistic']:.2f}")
        print(f"p值: {distribution_stats['p_value']:.6f}")
        
        if distribution_stats['is_uniform']:
            print("✓ 均匀分布验证通过（p值 > 0.05）")
        else:
            print("⚠️  均匀分布验证未通过（p值 <= 0.05）")
        
        print(f"\n偏差最大的数字: {distribution_stats['max_diff_num']}")
        print(f"  期望频率: {distribution_stats['max_diff']['expected']:.2f}")
        print(f"  实际频率: {distribution_stats['max_diff']['observed']}")
        print(f"  偏差: {distribution_stats['max_diff']['diff']:.2f}")
        print(f"  偏差比例: {distribution_stats['max_diff']['diff_ratio']:.2%}")
        
        print(f"\n偏差最小的数字: {distribution_stats['min_diff_num']}")
        print(f"  期望频率: {distribution_stats['min_diff']['expected']:.2f}")
        print(f"  实际频率: {distribution_stats['min_diff']['observed']}")
        print(f"  偏差: {distribution_stats['min_diff']['diff']:.2f}")
        print(f"  偏差比例: {distribution_stats['min_diff']['diff_ratio']:.2%}")
    
    print()
    print("=" * 80)
    print("验证完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

