"""
Batch processing optimizations for improved throughput.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar('T')
R = TypeVar('R')


class BatchProcessor:
    """Optimized batch processing for multiple operations."""
    
    def __init__(self, batch_size: int = 100, max_concurrent: int = 10):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], R],
        batch_size: Optional[int] = None
    ) -> List[R]:
        """Process items in optimized batches."""
        batch_size = batch_size or self.batch_size
        results = []
        
        # Process in chunks
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for item in batch:
                async def process_with_limit(item):
                    async with self.semaphore:
                        if asyncio.iscoroutinefunction(processor):
                            return await processor(item)
                        else:
                            return processor(item)
                
                tasks.append(process_with_limit(item))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
        
        return results
    
    async def process_stream(
        self,
        items: List[T],
        processor: Callable[[T], R],
        callback: Optional[Callable[[R], None]] = None
    ):
        """Process items as a stream with optional callback."""
        queue = asyncio.Queue(maxsize=self.batch_size)
        
        async def producer():
            for item in items:
                await queue.put(item)
            await queue.put(None)  # Signal completion
        
        async def consumer():
            while True:
                item = await queue.get()
                if item is None:
                    break
                
                async with self.semaphore:
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        result = processor(item)
                    
                    if callback:
                        callback(result)
        
        # Run producer and multiple consumers
        consumers = [consumer() for _ in range(self.max_concurrent)]
        await asyncio.gather(producer(), *consumers)
    
    def optimize_batch_size(self, total_items: int, processing_time_ms: float) -> int:
        """Calculate optimal batch size based on workload."""
        # Heuristic: Balance between memory and parallelization
        if processing_time_ms < 10:
            # Fast operations - larger batches
            return min(1000, total_items)
        elif processing_time_ms < 100:
            # Medium operations
            return min(100, total_items)
        else:
            # Slow operations - smaller batches for better responsiveness
            return min(10, total_items)