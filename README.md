# LangGraph Azure SQL Database Checkpoint Saver

A persistent checkpoint storage implementation for [LangGraph](https://github.com/langchain-ai/langgraph) applications using Azure SQL Database as the backend.

## Features

- **Persistent checkpoint storage** for LangGraph conversation state
- **Both sync and async** operations supported
- **Azure SQL Database** integration with Azure Active Directory authentication
- **Thread-based conversation management** with checkpoint history
- **Flexible configuration** with connection strings or individual parameters
- **Automatic table creation** and schema management

## Installation

### Basic Installation

```bash
pip install langgraph-azure-sql-db-checkpoint
```

### For Async Support

```bash
pip install langgraph-azure-sql-db-checkpoint[async]
```

### Development Installation

```bash
pip install langgraph-azure-sql-db-checkpoint[dev]
```

## Prerequisites

1. **Azure SQL Database** - You'll need an Azure SQL Database instance
2. **ODBC Driver** - Install "ODBC Driver 18 for SQL Server" on your system
3. **Authentication** - Either Azure AD authentication or SQL username/password

## Quick Start

### Synchronous Usage

```python
from langgraph_azure_sql_db_checkpoint import AzureSQLCheckpointSaver
from langgraph.graph import StateGraph

# Initialize with Azure AD authentication (recommended)
checkpointer = AzureSQLCheckpointSaver(
    server="your-server.database.windows.net",
    database="your-database"
)

# Setup the database table
checkpointer.setup()

# Use with LangGraph
app = StateGraph(YourStateClass)
# ... define your graph ...
app = app.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "user123"}}
result = app.invoke({"input": "Hello"}, config)
```

### Asynchronous Usage

```python
from langgraph_azure_sql_db_checkpoint import AsyncAzureSQLCheckpointSaver
import asyncio

async def main():
    # Initialize async checkpointer
    checkpointer = AsyncAzureSQLCheckpointSaver(
        server="your-server.database.windows.net",
        database="your-database"
    )
    
    # Setup the database table
    await checkpointer.asetup()
    
    # Use with async LangGraph
    app = StateGraph(YourStateClass)
    # ... define your graph ...
    app = app.compile(checkpointer=checkpointer)
    
    # Run with thread_id for persistence
    config = {"configurable": {"thread_id": "user123"}}
    result = await app.ainvoke({"input": "Hello"}, config)
    
    # Don't forget to cleanup
    await checkpointer.aclose()

asyncio.run(main())
```

## Authentication Options

### Azure Active Directory (Recommended)

```python
# Using Azure AD (default)
checkpointer = AzureSQLCheckpointSaver(
    server="your-server.database.windows.net",
    database="your-database",
    use_azure_auth=True  # This is the default
)
```

### SQL Authentication

```python
# Using SQL username/password
checkpointer = AzureSQLCheckpointSaver(
    server="your-server.database.windows.net",
    database="your-database",
    username="your-username",
    password="your-password",
    use_azure_auth=False
)
```

### Connection String

```python
# Using a full connection string
connection_string = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=your-server.database.windows.net;"
    "Database=your-database;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
)

checkpointer = AzureSQLCheckpointSaver(connection_string=connection_string)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `connection_string` | str | None | Full ODBC connection string |
| `server` | str | None | Azure SQL server name |
| `database` | str | None | Database name |
| `username` | str | None | SQL username (if not using Azure AD) |
| `password` | str | None | SQL password (if not using Azure AD) |
| `use_azure_auth` | bool | True | Use Azure AD authentication |
| `driver` | str | "ODBC Driver 18 for SQL Server" | ODBC driver name |
| `table_name` | str | "langgraph_checkpoints" | Name of the checkpoints table |
| `checkpoint_ns` | str | "checkpoint" | Checkpoint namespace prefix |

## Advanced Usage

### Managing Conversation History

```python
# Get conversation history for a thread
history = checkpointer.get_thread_history("user123", limit=10)

for checkpoint_tuple in history:
    print(f"Checkpoint: {checkpoint_tuple.checkpoint}")
    print(f"Metadata: {checkpoint_tuple.metadata}")

# Clear a conversation thread
checkpointer.clear_thread("user123")
```

### Custom Table Configuration

```python
checkpointer = AzureSQLCheckpointSaver(
    server="your-server.database.windows.net",
    database="your-database",
    table_name="my_custom_checkpoints",
    checkpoint_ns="my_app"
)
```

### Error Handling

```python
from sqlalchemy.exc import SQLAlchemyError

try:
    checkpointer = AzureSQLCheckpointSaver(
        server="your-server.database.windows.net",
        database="your-database"
    )
    checkpointer.setup()
except SQLAlchemyError as e:
    print(f"Database connection failed: {e}")
    # Handle connection errors
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Database Schema

The package automatically creates a table with the following schema:

```sql
CREATE TABLE langgraph_checkpoints (
    thread_id NVARCHAR(500) NOT NULL,
    checkpoint_id NVARCHAR(500) NOT NULL,
    parent_checkpoint_id NVARCHAR(500) NULL,
    checkpoint_data TEXT NOT NULL,
    metadata TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT GETUTCDATE(),
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

## Troubleshooting

### Common Issues

1. **ODBC Driver Not Found**
   ```
   Install "ODBC Driver 18 for SQL Server" from Microsoft
   ```

2. **Azure AD Authentication Failed**
   ```
   Ensure you're logged in with Azure CLI: az login
   Or use SQL authentication instead
   ```

3. **Connection Timeout**
   ```
   Check your Azure SQL firewall rules and connection string
   ```

### Debug Mode

```python
# Enable SQLAlchemy logging for debugging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/yourusername/langgraph-azure-sql-db-checkpoint.git
cd langgraph-azure-sql-db-checkpoint
pip install -e ".[dev]"
```

### Running Tests

The test file requires additional dependencies that are not part of the main package:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run the test file (requires Azure SQL connection and OpenAI API setup)
python tests/test_azure_sql_checkpoint.py
```

**Note**: The test file requires:
- `python-dotenv` for environment variable management
- `langchain-openai` for the Azure OpenAI integration example
- Proper environment variables set in a `.env` file:
  - `AZURE_SQL_CONN` - Azure SQL connection string
  - `AZURE_DEPLOYMENT_NAME` - Azure OpenAI deployment name
  - `AZURE_OPENAI_VERSION` - Azure OpenAI API version
  - `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL
  - `AZURE_OPENAI_API_KEY` - Azure OpenAI API key

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - The graph-based conversation framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Python SQL toolkit and ORM

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/yourusername/langgraph-azure-sql-db-checkpoint/issues) page.