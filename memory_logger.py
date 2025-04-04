import subprocess
import time
from collections import deque

def get_gpu_memory_usage():
    """
    使用 nvidia-smi 获取当前 GPU 显存使用量 (单位: MB)。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            memory_used = int(result.stdout.strip())
            return memory_used
        else:
            print("Error: Failed to retrieve GPU memory usage.")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def log_memory_usage():
    """
    每 10 秒记录一次显存使用量，并按要求输出 5 分钟和 20 分钟的峰值。
    """
    # 初始化数据结构
    memory_records = deque(maxlen=120)  # 最多保存 20 分钟的数据 (120 次记录)
    five_min_peak = 0
    twenty_min_peak = 0

    # 时间间隔设置
    record_interval = 10  # 每 10 秒记录一次
    five_min_interval = 5 * 60  # 5 分钟
    twenty_min_interval = 20 * 60  # 20 分钟

    # 计数器
    record_count = 0

    print("Starting GPU memory usage monitoring...")
    while True:
        # 获取显存使用量
        memory_used = get_gpu_memory_usage()
        if memory_used is not None:
            memory_records.append(memory_used)

            # 更新峰值
            five_min_peak = max(five_min_peak, memory_used)
            twenty_min_peak = max(twenty_min_peak, memory_used)

            # 每 5 分钟输出一次峰值
            if record_count > 0 and record_count % (five_min_interval // record_interval) == 0:
                print(f"[5-Minute Peak] >>> {five_min_peak} MB")
                five_min_peak = 0  # 重置 5 分钟峰值

            # 每 20 分钟输出一次峰值
            if record_count > 0 and record_count % (twenty_min_interval // record_interval) == 0:
                print(f"[20-Minute Peak] >>>>>>>>>>>>>>>>>>>>>> {twenty_min_peak} MB")
                twenty_min_peak = 0  # 重置 20 分钟峰值

        # 等待 10 秒
        time.sleep(record_interval)
        record_count += 1

if __name__ == "__main__":
    log_memory_usage()