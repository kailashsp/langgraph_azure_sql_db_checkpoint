import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph_azure_sql_db_checkpoint import AzureSQLCheckpointSaver, AsyncAzureSQLCheckpointSaver

load_dotenv()
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.1,
    max_retries=2,
    streaming=True,
)


# Initialize AzureSQLCheckpointSaver with connection string from environment
memory = AzureSQLCheckpointSaver(connection_string=os.getenv("AZURE_SQL_CONN"))
async_memory = AsyncAzureSQLCheckpointSaver(connection_string=os.getenv("AZURE_SQL_CONN"))

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=memory)

def test_sync_checkpoint():
    """Test the synchronous functionality of AzureSQLCheckpointSaver"""
    print("--- Starting Sync Test ---")
    
    config = {"configurable": {"thread_id": "3"}}

    # memory.delete(config)
    input_message = {"type": "user", "content": "hi! I'm bob"}
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # input_message = {"type": "user", "content": "what's my name?"}
    # for chunk in graph.stream(
    #     {"messages": [input_message]}, config, stream_mode="values"
    # ):
    #    chunk["messages"][-1].pretty_print()

    input_message = {"type": "user", "content": "I live in Pune?"}
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # input_message = {"type": "user", "content": "Tell me history of my place?"}
    # for chunk in graph.stream(
    #     {"messages": [input_message]}, config, stream_mode="values"
    # ):
    #    chunk["messages"][-1].pretty_print()
    
    print("--- Sync Test Completed ---")


async def async_call_model(state: MessagesState):
    """Async version of the call_model function"""
    response = await model.ainvoke(state["messages"])
    return {"messages": response}


async def test_async_checkpoint():
    """Test the async functionality of AzureSQLCheckpointSaver"""
    print("\n--- Starting Async Test ---")

    # Create async version of the graph
    async_builder = StateGraph(MessagesState)
    async_builder.add_node("call_model", async_call_model)
    async_builder.add_edge(START, "call_model")
    async_graph = async_builder.compile(checkpointer=async_memory)

    async_config = {"configurable": {"thread_id": "async_test_4"}}

    # Test first message
    input_message = {"type": "user", "content": "hi! I'm Alice"}
    async for chunk in async_graph.astream(
        {"messages": [input_message]}, async_config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Test second message to verify memory persistence
    input_message = {"type": "user", "content": "What's my name?"}
    async for chunk in async_graph.astream(
        {"messages": [input_message]}, async_config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    print("--- Async Test Completed ---")


if __name__ == "__main__":
    # Run the synchronous test
    test_sync_checkpoint()
    
    # Run the async test
    asyncio.run(test_async_checkpoint())
