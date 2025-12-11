# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-11

### Added
- Initial release of LangGraph Azure SQL Database Checkpoint Saver
- Synchronous checkpoint saver (`AzureSQLCheckpointSaver`)
- Asynchronous checkpoint saver (`AsyncAzureSQLCheckpointSaver`)
- Support for Azure Active Directory authentication
- Support for SQL Server authentication
- Automatic table creation and schema management
- Thread-based conversation management
- Checkpoint history functionality
- Comprehensive documentation and examples
- Support for Python 3.8+

### Features
- Persistent checkpoint storage in Azure SQL Database
- Connection pooling for efficient database usage
- Flexible configuration options
- Error handling and logging
- Compatible with LangGraph 0.2.0+