"""
真随机数生成模块
支持通过多种方式生成真随机数：
1. /dev/random (阻塞式，高质量随机数)
2. /dev/urandom (非阻塞式，快速)
3. TRNG设备（如果可用）
4. 硬件噪声源（如果可用）
"""

import os
import sys
from typing import Optional


class TrueRandomGenerator:
    """真随机数生成器"""
    
    def __init__(self, use_blocking: bool = True, fallback_to_pseudo: bool = True, silent: bool = False):
        """
        初始化真随机数生成器
        
        Args:
            use_blocking: 是否使用阻塞式/dev/random（更安全但可能较慢）
            fallback_to_pseudo: 如果真随机不可用，是否回退到伪随机
            silent: 是否静默模式（不打印初始化信息）
        """
        self.use_blocking = use_blocking
        self.fallback_to_pseudo = fallback_to_pseudo
        self._random_device = None
        self._available = False
        
        # 尝试打开随机设备
        self._init_random_device(silent=silent)
    
    def _init_random_device(self, silent: bool = False):
        """
        初始化随机设备
        
        Args:
            silent: 是否静默模式（不打印信息）
        """
        error_msg = None
        
        if sys.platform == 'linux':
            # Linux系统，尝试使用/dev/random或/dev/urandom
            if self.use_blocking:
                device_path = '/dev/random'
            else:
                device_path = '/dev/urandom'
            
            try:
                # 尝试打开设备
                self._random_device = open(device_path, 'rb')
                self._available = True
                if not silent:
                    print(f"✓ 真随机数生成器已初始化: 使用 {device_path}")
            except (IOError, OSError) as e:
                error_msg = f"无法打开 {device_path}: {e}"
                if not silent:
                    print(f"⚠️  {error_msg}")
                self._available = False
                if not self.fallback_to_pseudo:
                    # 不启用回退时，抛出错误
                    error = RuntimeError(
                        f"❌ 真随机数生成器初始化失败: {error_msg}\n"
                        f"   请检查系统是否支持真随机数生成，或设置 fallback_to_pseudo=True"
                    )
                    print(f"❌ 错误: {error}", file=sys.stderr)
                    raise error
                elif not silent:
                    print("  将回退到伪随机数生成器")
        else:
            # 非Linux系统，尝试使用os.urandom
            try:
                # 测试os.urandom是否可用
                os.urandom(1)
                self._available = True
                if not silent:
                    print("✓ 真随机数生成器已初始化: 使用 os.urandom")
            except Exception as e:
                error_msg = f"无法使用 os.urandom: {e}"
                if not silent:
                    print(f"⚠️  {error_msg}")
                self._available = False
                if not self.fallback_to_pseudo:
                    # 不启用回退时，抛出错误
                    error = RuntimeError(
                        f"❌ 真随机数生成器初始化失败: {error_msg}\n"
                        f"   请检查系统是否支持真随机数生成，或设置 fallback_to_pseudo=True"
                    )
                    print(f"❌ 错误: {error}", file=sys.stderr)
                    raise error
                elif not silent:
                    print("  将回退到伪随机数生成器")
    
    def _read_random_bytes(self, n: int) -> bytes:
        """
        从随机设备读取指定字节数
        
        Args:
            n: 要读取的字节数
            
        Returns:
            随机字节
        """
        if self._random_device:
            return self._random_device.read(n)
        elif sys.platform != 'linux':
            # 非Linux系统使用os.urandom
            return os.urandom(n)
        else:
            raise RuntimeError("真随机数生成器不可用")
    
    def random_int(self, min_val: int, max_val: int) -> int:
        """
        生成指定范围内的真随机整数（优化版，确保均匀分布）
        
        Args:
            min_val: 最小值（包含）
            max_val: 最大值（包含）
            
        Returns:
            真随机整数
        """
        if not self._available:
            if self.fallback_to_pseudo:
                import random
                return random.randint(min_val, max_val)
            else:
                error = RuntimeError(
                    "❌ 真随机数生成器不可用且未启用回退\n"
                    "   请检查系统是否支持真随机数生成，或设置 fallback_to_pseudo=True"
                )
                print(f"❌ 错误: {error}", file=sys.stderr)
                raise error
        
        # 计算范围大小
        range_size = max_val - min_val + 1
        if range_size <= 0:
            raise ValueError(f"无效的范围: [{min_val}, {max_val}]")
        
        # 使用拒绝采样确保完全均匀分布
        # 计算需要的位数和字节数
        bits_needed = (range_size - 1).bit_length()
        bytes_needed = max(1, (bits_needed + 7) // 8)  # 至少1字节
        
        # 计算最大随机值
        max_random_value = (256 ** bytes_needed) - 1
        
        # 计算最大可接受值（避免模运算偏差）
        # 只接受那些不会导致不均匀的值
        max_acceptable = max_random_value - (max_random_value % range_size)
        
        # 拒绝采样循环，确保完全均匀分布
        max_attempts = 10000  # 增加最大尝试次数
        attempts = 0
        
        while attempts < max_attempts:
            random_bytes = self._read_random_bytes(bytes_needed)
            random_int_val = int.from_bytes(random_bytes, byteorder='big')
            
            # 只接受在均匀范围内的值
            if random_int_val < max_acceptable:
                return min_val + (random_int_val % range_size)
            
            attempts += 1
        
        # 如果拒绝采样失败太多次（极罕见情况），使用模运算
        # 这种情况理论上不应该发生，但作为兜底方案
        random_bytes = self._read_random_bytes(bytes_needed)
        random_int_val = int.from_bytes(random_bytes, byteorder='big')
        return min_val + (random_int_val % range_size)
    
    def random_float(self) -> float:
        """
        生成[0.0, 1.0)范围内的真随机浮点数
        
        Returns:
            真随机浮点数
        """
        if not self._available:
            if self.fallback_to_pseudo:
                import random
                return random.random()
            else:
                error = RuntimeError(
                    "❌ 真随机数生成器不可用且未启用回退\n"
                    "   请检查系统是否支持真随机数生成，或设置 fallback_to_pseudo=True"
                )
                print(f"❌ 错误: {error}", file=sys.stderr)
                raise error
        
        # 读取4字节，转换为浮点数
        random_bytes = self._read_random_bytes(4)
        random_int = int.from_bytes(random_bytes, byteorder='big')
        # 转换为[0, 1)范围的浮点数
        return random_int / (2 ** 32)
    
    def shuffle(self, sequence: list) -> list:
        """
        使用真随机数打乱序列（Fisher-Yates洗牌算法）
        
        Args:
            sequence: 要打乱的序列
            
        Returns:
            打乱后的新列表
        """
        if not self._available:
            if self.fallback_to_pseudo:
                import random
                shuffled = sequence.copy()
                random.shuffle(shuffled)
                return shuffled
            else:
                error = RuntimeError(
                    "❌ 真随机数生成器不可用且未启用回退\n"
                    "   请检查系统是否支持真随机数生成，或设置 fallback_to_pseudo=True"
                )
                print(f"❌ 错误: {error}", file=sys.stderr)
                raise error
        
        shuffled = sequence.copy()
        n = len(shuffled)
        
        # Fisher-Yates洗牌算法
        for i in range(n - 1, 0, -1):
            j = self.random_int(0, i)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        
        return shuffled
    
    def sample(self, population: list, k: int) -> list:
        """
        从总体中随机选择k个不重复的元素（使用真随机）
        
        Args:
            population: 总体列表
            k: 要选择的元素数量
            
        Returns:
            选择的元素列表
        """
        if not self._available:
            if self.fallback_to_pseudo:
                import random
                return random.sample(population, k)
            else:
                error = RuntimeError(
                    "❌ 真随机数生成器不可用且未启用回退\n"
                    "   请检查系统是否支持真随机数生成，或设置 fallback_to_pseudo=True"
                )
                print(f"❌ 错误: {error}", file=sys.stderr)
                raise error
        
        if k > len(population):
            raise ValueError(f"k ({k}) 不能大于总体大小 ({len(population)})")
        
        if k == 0:
            return []
        
        # 使用Fisher-Yates算法选择k个元素
        population_copy = population.copy()
        selected = []
        
        for i in range(k):
            # 从剩余元素中随机选择一个
            j = self.random_int(0, len(population_copy) - 1)
            selected.append(population_copy.pop(j))
        
        return selected
    
    def is_available(self) -> bool:
        """检查真随机数生成器是否可用"""
        return self._available
    
    def close(self):
        """关闭随机设备"""
        if self._random_device:
            self._random_device.close()
            self._random_device = None
            self._available = False
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局真随机数生成器实例（延迟初始化）
_global_true_random: Optional[TrueRandomGenerator] = None


def get_true_random_generator(use_blocking: bool = True, fallback_to_pseudo: bool = True, silent: bool = False) -> TrueRandomGenerator:
    """
    获取全局真随机数生成器实例（单例模式）
    
    Args:
        use_blocking: 是否使用阻塞式/dev/random
        fallback_to_pseudo: 如果真随机不可用，是否回退到伪随机
        silent: 是否静默模式（不打印初始化信息）
        
    Returns:
        真随机数生成器实例
        
    Raises:
        RuntimeError: 当真随机不可用且fallback_to_pseudo=False时
    """
    global _global_true_random
    
    if _global_true_random is None:
        try:
            _global_true_random = TrueRandomGenerator(use_blocking, fallback_to_pseudo, silent=silent)
        except RuntimeError as e:
            # 如果初始化失败且不允许回退，重新抛出错误
            if not fallback_to_pseudo:
                print(f"❌ 获取真随机数生成器失败: {e}", file=sys.stderr)
                raise
            # 如果允许回退，创建一个回退到伪随机的实例
            _global_true_random = TrueRandomGenerator(use_blocking, fallback_to_pseudo=True, silent=silent)
    
    return _global_true_random


def reset_true_random_generator():
    """重置全局真随机数生成器"""
    global _global_true_random
    if _global_true_random:
        _global_true_random.close()
    _global_true_random = None


if __name__ == "__main__":
    """
    测试真随机数生成器
    生成10个1-100范围内的随机数（默认）
    可通过参数修改范围和数量
    """
    import sys
    
    # 解析命令行参数
    use_blocking = '--blocking' in sys.argv or '-b' in sys.argv
    min_val = 1
    max_val = 100
    count = 10
    
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
    
    # 解析数量参数
    if '--count' in sys.argv or '-n' in sys.argv:
        arg = '--count' if '--count' in sys.argv else '-n'
        idx = sys.argv.index(arg)
        if idx + 1 < len(sys.argv):
            try:
                count = int(sys.argv[idx + 1])
            except ValueError:
                pass
    
    print("=" * 60)
    print("真随机数生成器测试")
    print("=" * 60)
    
    # 创建真随机数生成器
    true_random = TrueRandomGenerator(
        use_blocking=use_blocking,
        fallback_to_pseudo=True
    )
    
    if not true_random.is_available():
        print("❌ 真随机数生成器不可用，退出")
        sys.exit(1)
    
    # 生成随机数
    print(f"\n生成{count}个随机数（范围: {min_val}-{max_val}）:")
    random_numbers = []
    for i in range(count):
        num = true_random.random_int(min_val, max_val)
        random_numbers.append(num)
        if count <= 20:  # 只在小数量时打印每个数字
            print(f"  随机数 {i+1}: {num}")
        elif (i+1) % (count // 10) == 0:  # 大数量时只打印进度
            print(f"  进度: {i+1}/{count}")
    
    print(f"\n生成的随机数列表: {random_numbers[:20]}{'...' if len(random_numbers) > 20 else ''}")
    print(f"随机数范围: [{min(random_numbers)}, {max(random_numbers)}]")
    print(f"平均值: {sum(random_numbers) / len(random_numbers):.2f}")
    
    # 关闭生成器
    true_random.close()
    print("\n✓ 测试完成")

