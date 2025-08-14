"""
Database connection pool optimization for Neo4j.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase, AsyncSession


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    active_connections: int = 0
    idle_connections: int = 0
    total_created: int = 0
    total_destroyed: int = 0
    wait_time_ms: float = 0
    queries_executed: int = 0
    errors: int = 0


class ConnectionPoolManager:
    """Optimized connection pool manager for Neo4j."""
    
    def __init__(
        self,
        uri: str,
        auth: tuple,
        max_connections: int = 50,
        min_connections: int = 10,
        connection_timeout: float = 30.0,
        max_connection_lifetime: float = 3600.0
    ):
        self.uri = uri
        self.auth = auth
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.max_connection_lifetime = max_connection_lifetime
        
        self.driver = None
        self.stats = ConnectionStats()
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        self._initialize_lock = asyncio.Lock()
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize connection pool."""
        async with self._initialize_lock:
            if self._is_initialized:
                return
            
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=self.auth,
                max_connection_pool_size=self.max_connections,
                connection_timeout=self.connection_timeout,
                max_connection_lifetime=self.max_connection_lifetime,
                connection_acquisition_timeout=self.connection_timeout,
                # Performance optimizations
                encrypted=False,  # Disable encryption for local connections
                trust=None,
                resolver=None,
                database=None,
                fetch_size=1000,  # Increase fetch size
                liveness_check_timeout=None
            )
            
            # Pre-warm connection pool
            await self._warm_pool()
            self._is_initialized = True
    
    async def _warm_pool(self):
        """Pre-warm connection pool with minimum connections."""
        tasks = []
        for _ in range(self.min_connections):
            async def warm_connection():
                async with self.driver.session() as session:
                    await session.run("RETURN 1")
            tasks.append(warm_connection())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats.total_created = self.min_connections
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict]:
        """Execute query with connection pool optimization."""
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        async with self.connection_semaphore:
            wait_time = time.perf_counter() - start_time
            self.stats.wait_time_ms = wait_time * 1000
            
            try:
                async with self.driver.session(database=database) as session:
                    self.stats.active_connections += 1
                    
                    result = await session.run(query, parameters or {})
                    records = [dict(record) async for record in result]
                    
                    self.stats.queries_executed += 1
                    self.stats.active_connections -= 1
                    
                    return records
                    
            except Exception as e:
                self.stats.errors += 1
                raise
    
    async def execute_batch(
        self,
        queries: List[Tuple[str, Dict[str, Any]]],
        database: Optional[str] = None
    ) -> List[List[Dict]]:
        """Execute multiple queries in a batch."""
        if not self._is_initialized:
            await self.initialize()
        
        async def execute_single(query, params):
            return await self.execute_query(query, params, database)
        
        tasks = [execute_single(q, p) for q, p in queries]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_transaction(
        self,
        transaction_func,
        database: Optional[str] = None
    ):
        """Execute a transaction with proper isolation."""
        if not self._is_initialized:
            await self.initialize()
        
        async with self.driver.session(database=database) as session:
            async with session.begin_transaction() as tx:
                try:
                    result = await transaction_func(tx)
                    await tx.commit()
                    return result
                except Exception as e:
                    await tx.rollback()
                    self.stats.errors += 1
                    raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "active_connections": self.stats.active_connections,
            "idle_connections": self.max_connections - self.stats.active_connections,
            "total_created": self.stats.total_created,
            "total_destroyed": self.stats.total_destroyed,
            "avg_wait_time_ms": self.stats.wait_time_ms,
            "queries_executed": self.stats.queries_executed,
            "errors": self.stats.errors,
            "error_rate": self.stats.errors / self.stats.queries_executed if self.stats.queries_executed > 0 else 0
        }
    
    async def health_check(self) -> bool:
        """Check connection pool health."""
        try:
            result = await self.execute_query("RETURN 1 as health")
            return len(result) > 0 and result[0].get("health") == 1
        except:
            return False
    
    async def cleanup(self):
        """Clean up connection pool."""
        if self.driver:
            await self.driver.close()
            self._is_initialized = False