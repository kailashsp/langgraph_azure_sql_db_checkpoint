"""LangGraph Azure SQL Database Checkpoint Saver.

This package provides persistent checkpoint storage for LangGraph applications
using Azure SQL Database as the backend. It supports both synchronous and
asynchronous operations.

Example:
    Basic usage:
    ```python
    from langgraph_azure_sql_db_checkpoint import AzureSQLCheckpointSaver

    # Using Azure AD authentication
    checkpointer = AzureSQLCheckpointSaver(
        server="your-server.database.windows.net",
        database="your-database"
    )

    # Setup the database table
    checkpointer.setup()
    ```

    Async usage:
    ```python
    from langgraph_azure_sql_db_checkpoint import AsyncAzureSQLCheckpointSaver

    checkpointer = AsyncAzureSQLCheckpointSaver(
        server="your-server.database.windows.net", 
        database="your-database"
    )

    # Setup the database table
    await checkpointer.asetup()
    ```
"""

from .azure_sql_checkpoint import AsyncAzureSQLCheckpointSaver, AzureSQLCheckpointSaver

__version__ = "0.1.1"
__author__ = "Kailash S Prem"
__email__ = "premkailash060@gmail.com"

__all__ = [
    "AzureSQLCheckpointSaver",
    "AsyncAzureSQLCheckpointSaver",
]