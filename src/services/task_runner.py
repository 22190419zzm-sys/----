"""
轻量任务调度器：统一后台执行入口，避免 UI 线程阻塞。
默认基于 ThreadPoolExecutor，后续可切换 QThreadPool。
"""
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any


class TaskRunner:
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="worker")

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        """提交后台任务，返回 Future，便于 UI 绑定进度/完成回调。"""
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


# 默认全局实例，UI 可直接复用
runner = TaskRunner()

