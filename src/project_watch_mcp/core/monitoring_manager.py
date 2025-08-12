"""Monitoring manager for persistent background monitoring."""

import asyncio
import logging
import signal
from pathlib import Path

from neo4j import AsyncDriver

from ..repository_monitor import RepositoryMonitor

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Manages persistent monitoring of repositories."""

    _instances: dict[str, "MonitoringManager"] = {}

    def __init__(
        self,
        repo_path: Path,
        project_name: str,
        neo4j_driver: AsyncDriver,
        file_patterns: list[str] | None = None,
    ):
        """
        Initialize the monitoring manager.

        Args:
            repo_path: Path to the repository
            project_name: Project name for identification
            neo4j_driver: Neo4j driver for database operations
            file_patterns: File patterns to monitor
        """
        self.repo_path = repo_path
        self.project_name = project_name
        self.neo4j_driver = neo4j_driver
        self.file_patterns = file_patterns or ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx"]

        self.monitor: RepositoryMonitor | None = None
        self._monitoring_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Register this instance
        self._instances[project_name] = self

    @classmethod
    def get_instance(cls, project_name: str) -> "MonitoringManager | None":
        """Get an existing monitoring manager instance by project name."""
        return cls._instances.get(project_name)

    @classmethod
    def is_monitoring(cls, project_name: str) -> bool:
        """Check if a project is being monitored."""
        instance = cls.get_instance(project_name)
        return instance is not None and instance.is_running()

    def is_running(self) -> bool:
        """Check if monitoring is currently running."""
        return (
            self.monitor is not None
            and self.monitor.is_running
            and self._monitoring_task is not None
            and not self._monitoring_task.done()
        )

    async def start_persistent_monitoring(self) -> bool:
        """
        Start persistent monitoring that continues in the background.

        Returns:
            True if monitoring was started successfully
        """
        if self.is_running():
            logger.info(f"Monitoring already running for {self.project_name}")
            return True

        try:
            # Create monitor
            self.monitor = RepositoryMonitor(
                repo_path=self.repo_path,
                project_name=self.project_name,
                neo4j_driver=self.neo4j_driver,
                file_patterns=self.file_patterns,
            )

            # Start monitoring
            await self.monitor.start(daemon=True)

            # Create a task to keep monitoring running
            self._monitoring_task = asyncio.create_task(
                self._monitor_loop(),
                name=f"monitoring_manager_{self.project_name}"
            )

            logger.info(f"Started persistent monitoring for {self.project_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start monitoring for {self.project_name}: {e}")
            return False

    async def _monitor_loop(self):
        """Main monitoring loop that processes changes."""
        logger.debug(f"Monitoring loop started for {self.project_name}")

        try:
            while not self._shutdown_event.is_set():
                if not self.monitor or not self.monitor.is_running:
                    logger.warning(f"Monitor stopped unexpectedly for {self.project_name}")
                    break

                # Process any pending changes
                if self.monitor.has_pending_changes():
                    changes = await self.monitor.process_all_changes()
                    logger.debug(f"Processed {len(changes)} changes for {self.project_name}")

                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.debug(f"Monitoring loop cancelled for {self.project_name}")
        except Exception as e:
            logger.error(f"Error in monitoring loop for {self.project_name}: {e}")
        finally:
            logger.debug(f"Monitoring loop ended for {self.project_name}")

    async def stop(self):
        """Stop monitoring."""
        self._shutdown_event.set()

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        if self.monitor:
            await self.monitor.stop()
            self.monitor = None

        # Remove from instances
        self._instances.pop(self.project_name, None)

        logger.info(f"Stopped monitoring for {self.project_name}")

    @classmethod
    async def shutdown_all(cls):
        """Shutdown all monitoring instances."""
        for instance in list(cls._instances.values()):
            await instance.stop()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def handle_shutdown(sig, frame):
        logger.info(f"Received signal {sig}, shutting down monitors...")
        asyncio.create_task(MonitoringManager.shutdown_all())

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
