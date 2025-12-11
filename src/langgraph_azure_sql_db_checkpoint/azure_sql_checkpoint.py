"""Azure SQL checkpoint store for LangGraph."""

import hashlib
import json
import threading
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote_plus

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.mssql import NVARCHAR
from sqlalchemy.orm import sessionmaker


class AzureSQLCheckpointSaver(BaseCheckpointSaver):
    """Synchronous Azure SQL Server checkpoint saver for LangGraph.

    Stores conversation checkpoints and metadata in Azure SQL Database,
    allowing for persistent conversation state across sessions.

    This is the synchronous-only version. For async operations, use AsyncAzureSQLCheckpointSaver.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_azure_auth: bool = True,
        driver: str = "ODBC Driver 18 for SQL Server",
        table_name: str = "langgraph_checkpoints",
        checkpoint_ns: str = "checkpoint",
    ):
        """Initialize the Azure SQL Checkpoint Saver.

        Args:
            connection_string: Full Azure SQL connection string
            server: Azure SQL server name
            database: Database name
            username: SQL username
            password: SQL password
            use_azure_auth: Use Azure Active Directory authentication
            driver: ODBC driver name
            table_name: Name of the checkpoints table
            checkpoint_ns: Checkpoint namespace prefix
        """
        super().__init__()

        self.table_name = table_name
        self.checkpoint_ns = checkpoint_ns
        self.lock = threading.Lock()

        # Build connection string
        if connection_string:
            self.connection_string = connection_string
        else:
            if not server or not database:
                raise ValueError(
                    "Either connection_string or server+database must be provided"
                )

            if use_azure_auth:
                self.connection_string = (
                    f"Driver={{{driver}}};"
                    f"Server={server};"
                    f"Database={database};"
                    f"Trusted_Connection=yes;"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                )
            else:
                if not username or not password:
                    raise ValueError(
                        "Username and password required when not using Azure auth"
                    )
                self.connection_string = (
                    f"Driver={{{driver}}};"
                    f"Server={server};"
                    f"Database={database};"
                    f"UID={username};"
                    f"PWD={password};"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                )

        # Create SQLAlchemy engine with proper connection string encoding
        connection_string_encoded = quote_plus(self.connection_string)
        self.engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={connection_string_encoded}",
            pool_pre_ping=True,
            pool_recycle=300,
        )

        self.SessionLocal = sessionmaker(bind=self.engine)

        # Define table schema
        self.metadata = MetaData()
        self.checkpoints_table = Table(
            self.table_name,
            self.metadata,
            Column("thread_id", NVARCHAR(500), primary_key=True),
            Column("checkpoint_id", NVARCHAR(500), primary_key=True),
            Column("parent_checkpoint_id", NVARCHAR(500), nullable=True),
            Column("checkpoint_data", Text, nullable=False),
            Column("metadata", Text, nullable=True),
            Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
        )

    def setup(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        self.metadata.create_all(self.engine)

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(checkpoint, default=str, ensure_ascii=False)

    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """Deserialize checkpoint from JSON string."""
        return json.loads(data)

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(metadata, default=str, ensure_ascii=False)

    def _deserialize_metadata(self, data: Optional[str]) -> CheckpointMetadata:
        """Deserialize metadata from JSON string."""
        if data:
            return json.loads(data)
        return {}

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save a checkpoint to the database."""
        thread_id = config["configurable"]["thread_id"]
        # Generate a checkpoint ID if not provided
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            # Create a simple checkpoint ID based on content
            import hashlib

            checkpoint_str = json.dumps(checkpoint, sort_keys=True, default=str)
            checkpoint_id = hashlib.md5(checkpoint_str.encode()).hexdigest()

        parent_checkpoint_id = config["configurable"].get("parent_checkpoint_id")

        checkpoint_data = self._serialize_checkpoint(checkpoint)
        metadata_json = self._serialize_metadata(metadata)

        with self.SessionLocal() as session:
            try:
                # Check if checkpoint already exists
                existing = session.execute(
                    text(f"""
                        SELECT checkpoint_id FROM {self.table_name}
                        WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                    """),
                    {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                ).fetchone()

                if existing:
                    # Update existing checkpoint
                    session.execute(
                        text(f"""
                            UPDATE {self.table_name}
                            SET checkpoint_data = :checkpoint_data, metadata = :metadata
                            WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                        """),
                        {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint_id,
                            "checkpoint_data": checkpoint_data,
                            "metadata": metadata_json,
                        },
                    )
                else:
                    # Insert new checkpoint
                    session.execute(
                        text(f"""
                            INSERT INTO {self.table_name}
                            (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata, created_at)
                            VALUES (:thread_id, :checkpoint_id, :parent_checkpoint_id, :checkpoint_data, :metadata, GETUTCDATE())
                        """),
                        {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint_id,
                            "parent_checkpoint_id": parent_checkpoint_id,
                            "checkpoint_data": checkpoint_data,
                            "metadata": metadata_json,
                        },
                    )

                session.commit()

                # Return updated config
                return {
                    **config,
                    "configurable": {
                        **config["configurable"],
                        "checkpoint_id": checkpoint_id,
                    },
                }

            except Exception:
                session.rollback()
                raise

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with self.SessionLocal() as session:
            if checkpoint_id:
                # Get specific checkpoint
                result = session.execute(
                    text(f"""
                        SELECT checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata
                        FROM {self.table_name}
                        WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                    """),
                    {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                ).fetchone()
            else:
                # Get latest checkpoint for thread
                result = session.execute(
                    text(f"""
                        SELECT TOP 1 checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata
                        FROM {self.table_name}
                        WHERE thread_id = :thread_id
                        ORDER BY created_at DESC
                    """),
                    {"thread_id": thread_id},
                ).fetchone()

            if result:
                checkpoint = self._deserialize_checkpoint(result.checkpoint_data)
                metadata = self._deserialize_metadata(result.metadata)

                parent_config = None
                if result.parent_checkpoint_id:
                    parent_config = {
                        **config,
                        "configurable": {
                            **config["configurable"],
                            "checkpoint_id": result.parent_checkpoint_id,
                        },
                    }

                return CheckpointTuple(
                    config={
                        **config,
                        "configurable": {
                            **config["configurable"],
                            "checkpoint_id": result.checkpoint_id,
                        },
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )

            return None

    def list(
        self,
        config: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        thread_id = config["configurable"]["thread_id"]

        with self.SessionLocal() as session:
            query = f"""
                SELECT checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata
                FROM {self.table_name}
                WHERE thread_id = :thread_id
            """
            params = {"thread_id": thread_id}

            if before:
                query += " AND created_at < (SELECT created_at FROM {self.table_name} WHERE checkpoint_id = :before)"
                params["before"] = before

            query += " ORDER BY created_at DESC"

            if limit:
                query = query.replace("SELECT", f"SELECT TOP {limit}")

            results = session.execute(text(query), params).fetchall()

            for row in results:
                checkpoint = self._deserialize_checkpoint(row.checkpoint_data)
                metadata = self._deserialize_metadata(row.metadata)

                parent_config = None
                if row.parent_checkpoint_id:
                    parent_config = {
                        **config,
                        "configurable": {
                            **config["configurable"],
                            "checkpoint_id": row.parent_checkpoint_id,
                        },
                    }

                yield CheckpointTuple(
                    config={
                        **config,
                        "configurable": {
                            **config["configurable"],
                            "checkpoint_id": row.checkpoint_id,
                        },
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Put writes to the database (placeholder implementation)."""
        # This could be implemented to store intermediate writes
        # For now, we'll skip this as it's optional for basic functionality
        pass

    def clear_thread(self, thread_id: str) -> None:
        """Clear all checkpoints for a thread."""
        with self.SessionLocal() as session:
            try:
                session.execute(
                    text(f"DELETE FROM {self.table_name} WHERE thread_id = :thread_id"),
                    {"thread_id": thread_id},
                )
                session.commit()
            except Exception:
                session.rollback()
                raise

    def get_thread_history(
        self, thread_id: str, limit: int = 10
    ) -> List[CheckpointTuple]:
        """Get checkpoint history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return list(self.list(config, limit=limit))

    def close(self) -> None:
        """Close all database connections and dispose engines."""
        if self.engine:
            self.engine.dispose()

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            if self.engine:
                self.engine.dispose()
        except Exception:
            pass  # Ignore cleanup errors during destruction


class AsyncAzureSQLCheckpointSaver(BaseCheckpointSaver):
    """Asynchronous Azure SQL Server checkpoint saver for LangGraph.

    Stores conversation checkpoints and metadata in Azure SQL Database,
    allowing for persistent conversation state across sessions.

    This is the async-only version. For sync operations, use AzureSQLCheckpointSaver.
    Requires 'aioodbc' package to be installed.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_azure_auth: bool = True,
        driver: str = "ODBC Driver 18 for SQL Server",
        table_name: str = "langgraph_checkpoints",
        checkpoint_ns: str = "checkpoint",
    ):
        """Initialize the Azure SQL Checkpoint Saver (async version).

        Args:
            connection_string: Full Azure SQL connection string
            server: Azure SQL server name
            database: Database name
            username: SQL username
            password: SQL password
            use_azure_auth: Use Azure Active Directory authentication
            driver: ODBC driver name
            table_name: Name of the checkpoints table
            checkpoint_ns: Checkpoint namespace prefix

        Raises:
            RuntimeError: If aioodbc is not installed
        """
        super().__init__()

        self.table_name = table_name
        self.checkpoint_ns = checkpoint_ns

        # Build connection string
        if connection_string:
            self.connection_string = connection_string
        else:
            if not server or not database:
                raise ValueError(
                    "Either connection_string or server+database must be provided"
                )

            if use_azure_auth:
                self.connection_string = (
                    f"Driver={{{driver}}};"
                    f"Server={server};"
                    f"Database={database};"
                    f"Trusted_Connection=yes;"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                )
            else:
                if not username or not password:
                    raise ValueError(
                        "Username and password required when not using Azure auth"
                    )
                self.connection_string = (
                    f"Driver={{{driver}}};"
                    f"Server={server};"
                    f"Database={database};"
                    f"UID={username};"
                    f"PWD={password};"
                    f"Encrypt=yes;"
                    f"TrustServerCertificate=no;"
                )

        # Create async SQLAlchemy engine
        try:
            from sqlalchemy.ext.asyncio import (
                create_async_engine,
                AsyncSession,
                async_sessionmaker,
            )

            # Check if aioodbc is available
            try:
                import aioodbc  # noqa: F401
            except ImportError:
                raise ImportError("aioodbc not installed")

            connection_string_encoded = quote_plus(self.connection_string)
            self.async_engine = create_async_engine(
                f"mssql+aioodbc:///?odbc_connect={connection_string_encoded}",
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False,
                pool_size=5,
                max_overflow=10,
            )
            self.AsyncSessionLocal = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                "Async checkpoint saver requires 'aioodbc' package. "
                "Install it with: pip install aioodbc"
            ) from e

        # Define table schema
        self.metadata = MetaData()
        self.checkpoints_table = Table(
            self.table_name,
            self.metadata,
            Column("thread_id", NVARCHAR(500), primary_key=True),
            Column("checkpoint_id", NVARCHAR(500), primary_key=True),
            Column("parent_checkpoint_id", NVARCHAR(500), nullable=True),
            Column("checkpoint_data", Text, nullable=False),
            Column("metadata", Text, nullable=True),
            Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
        )

    async def asetup(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(self.metadata.create_all)

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(checkpoint, default=str, ensure_ascii=False)

    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """Deserialize checkpoint from JSON string."""
        return json.loads(data)

    def _serialize_metadata(self, metadata: CheckpointMetadata) -> str:
        """Serialize metadata to JSON string."""
        return json.dumps(metadata, default=str, ensure_ascii=False)

    def _deserialize_metadata(self, data: Optional[str]) -> CheckpointMetadata:
        """Deserialize metadata from JSON string."""
        if data:
            return json.loads(data)
        return {}

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save a checkpoint to the database."""
        thread_id = config["configurable"]["thread_id"]
        # Generate a checkpoint ID if not provided
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            # Create a simple checkpoint ID based on content
            checkpoint_str = json.dumps(checkpoint, sort_keys=True, default=str)
            checkpoint_id = hashlib.md5(checkpoint_str.encode()).hexdigest()

        parent_checkpoint_id = config["configurable"].get("parent_checkpoint_id")

        checkpoint_data = self._serialize_checkpoint(checkpoint)
        metadata_json = self._serialize_metadata(metadata)

        async with self.AsyncSessionLocal() as session:
            try:
                # Check if checkpoint already exists
                result = await session.execute(
                    text(f"""
                        SELECT checkpoint_id FROM {self.table_name}
                        WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                    """),
                    {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                )
                existing = result.fetchone()

                if existing:
                    # Update existing checkpoint
                    await session.execute(
                        text(f"""
                            UPDATE {self.table_name}
                            SET checkpoint_data = :checkpoint_data, metadata = :metadata
                            WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                        """),
                        {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint_id,
                            "checkpoint_data": checkpoint_data,
                            "metadata": metadata_json,
                        },
                    )
                else:
                    # Insert new checkpoint
                    await session.execute(
                        text(f"""
                            INSERT INTO {self.table_name}
                            (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata, created_at)
                            VALUES (:thread_id, :checkpoint_id, :parent_checkpoint_id, :checkpoint_data, :metadata, GETUTCDATE())
                        """),
                        {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint_id,
                            "parent_checkpoint_id": parent_checkpoint_id,
                            "checkpoint_data": checkpoint_data,
                            "metadata": metadata_json,
                        },
                    )

                await session.commit()

                # Return updated config
                return {
                    **config,
                    "configurable": {
                        **config["configurable"],
                        "checkpoint_id": checkpoint_id,
                    },
                }

            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        async with self.AsyncSessionLocal() as session:
            try:
                if checkpoint_id:
                    # Get specific checkpoint
                    result = await session.execute(
                        text(f"""
                            SELECT checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata
                            FROM {self.table_name}
                            WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                        """),
                        {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                    )
                else:
                    # Get latest checkpoint for thread
                    result = await session.execute(
                        text(f"""
                            SELECT TOP 1 checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata
                            FROM {self.table_name}
                            WHERE thread_id = :thread_id
                            ORDER BY created_at DESC
                        """),
                        {"thread_id": thread_id},
                    )

                row = result.fetchone()

                if row:
                    checkpoint = self._deserialize_checkpoint(row.checkpoint_data)
                    metadata = self._deserialize_metadata(row.metadata)

                    parent_config = None
                    if row.parent_checkpoint_id:
                        parent_config = {
                            **config,
                            "configurable": {
                                **config["configurable"],
                                "checkpoint_id": row.parent_checkpoint_id,
                            },
                        }

                    return CheckpointTuple(
                        config={
                            **config,
                            "configurable": {
                                **config["configurable"],
                                "checkpoint_id": row.checkpoint_id,
                            },
                        },
                        checkpoint=checkpoint,
                        metadata=metadata,
                        parent_config=parent_config,
                    )

                return None
            finally:
                await session.close()

    async def alist(
        self,
        config: Dict[str, Any],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        thread_id = config["configurable"]["thread_id"]

        async with self.AsyncSessionLocal() as session:
            try:
                query = f"""
                    SELECT checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata
                    FROM {self.table_name}
                    WHERE thread_id = :thread_id
                """
                params = {"thread_id": thread_id}

                if before:
                    query += " AND created_at < (SELECT created_at FROM {self.table_name} WHERE checkpoint_id = :before)"
                    params["before"] = before

                query += " ORDER BY created_at DESC"

                if limit:
                    query = query.replace("SELECT", f"SELECT TOP {limit}")

                result = await session.execute(text(query), params)
                results = result.fetchall()

                for row in results:
                    checkpoint = self._deserialize_checkpoint(row.checkpoint_data)
                    metadata = self._deserialize_metadata(row.metadata)

                    parent_config = None
                    if row.parent_checkpoint_id:
                        parent_config = {
                            **config,
                            "configurable": {
                                **config["configurable"],
                                "checkpoint_id": row.parent_checkpoint_id,
                            },
                        }

                    yield CheckpointTuple(
                        config={
                            **config,
                            "configurable": {
                                **config["configurable"],
                                "checkpoint_id": row.checkpoint_id,
                            },
                        },
                        checkpoint=checkpoint,
                        metadata=metadata,
                        parent_config=parent_config,
                    )
            finally:
                await session.close()

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Put writes to the database (placeholder implementation)."""
        # This could be implemented to store intermediate writes
        # For now, we'll skip this as it's optional for basic functionality
        pass

    async def aclear_thread(self, thread_id: str) -> None:
        """Clear all checkpoints for a thread."""
        async with self.AsyncSessionLocal() as session:
            try:
                await session.execute(
                    text(f"DELETE FROM {self.table_name} WHERE thread_id = :thread_id"),
                    {"thread_id": thread_id},
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def aget_thread_history(
        self, thread_id: str, limit: int = 10
    ) -> List[CheckpointTuple]:
        """Get checkpoint history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        results = []
        async for item in self.alist(config, limit=limit):
            results.append(item)
        return results

    async def aclose(self) -> None:
        """Close all database connections and dispose engines."""
        if self.async_engine:
            await self.async_engine.dispose()

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            # Note: Cannot call async dispose in __del__
            # Use aclose() explicitly before object destruction
            pass
        except Exception:
            pass  # Ignore cleanup errors during destruction

    # Sync methods should raise NotImplementedError to make it clear this is async-only
    def setup(self) -> None:
        """Not implemented. Use asetup() for async checkpoint saver."""
        raise NotImplementedError("Use asetup() for async checkpoint saver")

    def put(self, config, checkpoint, metadata, new_versions):
        """Not implemented. Use aput() for async checkpoint saver."""
        raise NotImplementedError("Use aput() for async checkpoint saver")

    def get_tuple(self, config):
        """Not implemented. Use aget_tuple() for async checkpoint saver."""
        raise NotImplementedError("Use aget_tuple() for async checkpoint saver")

    def list(self, config, **kwargs):
        """Not implemented. Use alist() for async checkpoint saver."""
        raise NotImplementedError("Use alist() for async checkpoint saver")

    def put_writes(self, config, writes, task_id):
        """Not implemented. Use aput_writes() for async checkpoint saver."""
        raise NotImplementedError("Use aput_writes() for async checkpoint saver")

    def clear_thread(self, thread_id):
        """Not implemented. Use aclear_thread() for async checkpoint saver."""
        raise NotImplementedError("Use aclear_thread() for async checkpoint saver")

    def get_thread_history(self, thread_id, limit=10):
        """Not implemented. Use aget_thread_history() for async checkpoint saver."""
        raise NotImplementedError(
            "Use aget_thread_history() for async checkpoint saver"
        )

    def close(self):
        """Not implemented. Use aclose() for async checkpoint saver."""
        raise NotImplementedError("Use aclose() for async checkpoint saver")
