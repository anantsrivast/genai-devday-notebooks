

# ================================
# Step 1: Setup prerequisites
# ================================
import os
import sys
import json
import datetime
from typing import List, Dict, Annotated

from pymongo import MongoClient

# Add parent directory to path to import from utils
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))
from utils import track_progress, set_env, create_index, check_index_ready, get_llm

# --- NEW (LTM): imports for long-term memory ---
# MongoDB long-term memory store (langgraph-store-mongodb)
from langgraph.store.mongodb.base import MongoDBStore, VectorIndexConfig  # LTM store
from langchain_voyageai import VoyageAIEmbeddings                        # LTM embeddings
from langmem import create_manage_memory_tool, create_search_memory_tool # LTM tools

# LangGraph core imports
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver

# LangChain / tools / LLM plumbing
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate as LCChatPromptTemplate
from pprint import pprint
import voyageai


# ================================
# MongoDB + environment setup
# ================================

# If you are using your own MongoDB Atlas cluster, use the connection string for your cluster here
MONGODB_URI = os.environ.get("MONGODB_URI")

# Initialize a MongoDB Python client
mongodb_client = MongoClient(MONGODB_URI)

# Check the connection to the server
mongodb_client.admin.command("ping")

# Track progress of key steps-- DO NOT CHANGE
track_progress("cluster_creation", "ai_agents_lab")

# Set the LLM provider and passkey provided by your workshop instructor
# NOTE: LLM_PROVIDER can be set to one of "aws"/ "microsoft" / "google"
LLM_PROVIDER = "aws"
PASSKEY = "replace-with-passkey"

# Obtain API keys from our AI model proxy and set them as environment variables-- DO NOT CHANGE
set_env([LLM_PROVIDER, "voyageai"], PASSKEY)


# --- NEW (LTM): create MongoDBStore for long-term memory ---

MEMORY_DB_NAME = "mongodb_genai_devday_agents"
MEMORY_COLLECTION_NAME = "agent_memories"

# Get a handle to the long-term memory collection
memory_db = mongodb_client[MEMORY_DB_NAME]
memory_collection = memory_db[MEMORY_COLLECTION_NAME]

# Optional vector index configuration for semantic memory search
memory_index_config = VectorIndexConfig(
    embed=VoyageAIEmbeddings(model="voyage-3"),  # reuse Voyage model
    dims=1024,                                   # match your doc embedding dims
    fields=["value.content"],                    # field that holds memory text
)

# MongoDBStore for long-term, cross-thread memory
memory_store = MongoDBStore(
    collection=memory_collection,
    index_config=memory_index_config,
)


# ================================
# Step 2: Import data into MongoDB
# ================================

# Do not change the values assigned to the variables below
# Database name
DB_NAME = "mongodb_genai_devday_agents"
# Name of the collection with full documents- used for summarization
FULL_COLLECTION_NAME = "mongodb_docs"
# Name of the collection for vector search- used for Q&A
VS_COLLECTION_NAME = "mongodb_docs_embeddings"
# Name of the vector search index
VS_INDEX_NAME = "vector_index"

# Connect to the `VS_COLLECTION_NAME` collection.
vs_collection = mongodb_client[DB_NAME][VS_COLLECTION_NAME]
# Connect to the `FULL_COLLECTION_NAME` collection.
full_collection = mongodb_client[DB_NAME][FULL_COLLECTION_NAME]

# Insert a dataset of MongoDB docs with embeddings into the `VS_COLLECTION_NAME` collection
with open(f"../data/{VS_COLLECTION_NAME}.json", "r") as data_file:
    json_data = data_file.read()

data = json.loads(json_data)

print(f"Deleting existing documents from the {VS_COLLECTION_NAME} collection.")
vs_collection.delete_many({})
vs_collection.insert_many(data)
print(
    f"{vs_collection.count_documents({})} documents ingested into the {VS_COLLECTION_NAME} collection."
)

# Insert a dataset of MongoDB documentation pages into the `FULL_COLLECTION_NAME` collection
with open(f"../data/{FULL_COLLECTION_NAME}.json", "r") as data_file:
    json_data = data_file.read()

data = json.loads(json_data)

print(f"Deleting existing documents from the {FULL_COLLECTION_NAME} collection.")
full_collection.delete_many({})
full_collection.insert_many(data)
print(
    f"{full_collection.count_documents({})} documents ingested into the {FULL_COLLECTION_NAME} collection."
)


# ================================
# Step 3: Create a vector search index
# ================================
# Create vector index definition specifying:
# path: Path to the embeddings field
# numDimensions: Number of embedding dimensions- depends on the embedding model used
# similarity: Similarity metric. One of cosine, euclidean, dotProduct.
model = {
    "name": VS_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

# Use the `create_index` function from the `utils` module to create a vector search index
create_index(vs_collection, VS_INDEX_NAME, model)

# Use the `check_index_ready` function from the `utils` module to verify that the index was created
check_index_ready(vs_collection, VS_INDEX_NAME)

# Track progress of key steps-- DO NOT CHANGE
track_progress("vs_index_creation", "ai_agents_lab")


# ================================
# Step 4: Create agent tools
# ================================

# Initialize the Voyage AI client
vo = voyageai.Client()


# -------- Vector Search helper --------
def get_embeddings(query: str) -> List[float]:
    """
    Get embeddings for an input query.

    Args:
        query (str): Query string

    Returns:
        List[float]: Embedding of the query string
    """
    embds_obj = vo.contextualized_embed(
        inputs=[[query]],              # list of list-of-chunks; here single chunk
        model="voyage-context-3",
        input_type="query",
    )
    embeddings = embds_obj.embeddings[0]
    return embeddings


@tool
def get_information_for_question_answering(user_query: str) -> str:
    """
    Retrieve information using vector search to answer a user query.

    Args:
    user_query (str): The user's query string.

    Returns:
    str: The retrieved information formatted as a string.
    """

    # Generate embeddings for the `user_query`
    query_embedding = get_embeddings(user_query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": VS_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "body": 1,
                "vectorSearchScore": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(vs_collection.aggregate(pipeline))
    context = "\n\n".join([doc.get("body", "") for doc in results])
    return context


@tool
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Retrieve page content based on provided title.

    Args:
    user_query (str): The user's query string i.e. title of the documentation page.

    Returns:
    str: The content of the page.
    """
    query = {"title": user_query}
    projection = {"_id": 0, "body": 1}
    document = full_collection.find_one(query, projection)

    if document:
        return document["body"]
    else:
        return "Document not found"


# --- NEW (LTM): LangMem tools for long-term memory ---
manage_memory_tool = create_manage_memory_tool(
    namespace=("memories", "{user_id}"),  # user_id comes from graph config
)

search_memory_tool = create_search_memory_tool(
    namespace=("memories", "{user_id}"),
)

# Create the list of tools (doc tools + memory tools)
tools = [
    get_information_for_question_answering,
    get_page_content_for_summarization,
    search_memory_tool,   # long-term memory read
    manage_memory_tool,   # long-term memory write/update/delete
]


# -------- Test tools (optional) --------
get_information_for_question_answering.invoke(
    "What are some best practices for data backups in MongoDB?"
)
get_page_content_for_summarization.invoke("Create a MongoDB Deployment")


# ================================
# Step 5: Define graph state
# ================================

# Define the graph state
# We are only tracking chat messages but you can track other attributes as well
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


# ================================
# Step 6: Instantiate the LLM
# ================================
# Obtain the Langchain LLM object using the `get_llm` function from the `utils` module.
llm = get_llm(LLM_PROVIDER)

# Create a Chain-of-Thought (CoT) prompt template for the agent.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "You are a helpful AI assistant."
            " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
            " You ALSO have long-term memory tools that let you store and retrieve important information"
            " across conversations (for example, user roles, preferences, and high-level summaries of work)."
            " Use the memory search tool when prior interactions could help answer the current question"
            " (e.g., 'as we discussed before', 'what did I say about backups?')."
            " Use the memory management tool ONLY for stable, long-lived facts or compact summaries;"
            " do NOT store trivial or one-off details."
            " Think step-by-step and use these tools to get the information required to answer the user query."
            " Do not re-run tools unless absolutely necessary."
            " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
            " You have access to the following tools: {tool_names}."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Fill in the prompt template with the tool names
prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

# Bind tools to the LLM
bind_tools = llm.bind_tools(tools)

# Chain prompt → tool-augmented LLM
llm_with_tools = prompt | bind_tools

# Test that the LLM is making the right tool calls
llm_with_tools.invoke(
    ["Give me a summary of the page titled Create a MongoDB Deployment."]
).tool_calls
llm_with_tools.invoke(
    ["What are some best practices for data backups in MongoDB?"]
).tool_calls


# ================================
# Step 7: Define graph nodes
# ================================
def agent(state: GraphState) -> Dict[str, List]:
    """
    Agent node

    Args:
        state (GraphState): Graph state

    Returns:
        Dict[str, List]: Updates to messages
    """
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


# Create a map of tool name to tool object
tools_by_name = {tool.name: tool for tool in tools}
pprint(tools_by_name)


def tool_node(state: GraphState) -> Dict[str, List]:
    """
    Tool node

    Args:
        state (GraphState): Graph state

    Returns:
        Dict[str, List]: Updates to messages
    """
    result: List[ToolMessage] = []
    # Get the list of tool calls from messages (latest AI message)
    tool_calls = state["messages"][-1].tool_calls

    # Iterate through `tool_calls`
    for tool_call in tool_calls:
        tool_obj = tools_by_name[tool_call["name"]]
        observation = tool_obj.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}


# ================================
# Step 8: Define conditional edges
# ================================
def route_tools(state: GraphState):
    """
    Use in the conditional_edge to route to the tool node if the last message
    has tool calls. Otherwise, route to the end.
    """
    messages = state.get("messages", [])
    if len(messages) == 0:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    ai_message = messages[-1]

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# ================================
# Step 9: Build the graph
# ================================
from IPython.display import display

# Instantiate the graph
graph = StateGraph(GraphState)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")

graph.add_conditional_edges(
    "agent",
    route_tools,
    {
        "tools": "tools",
        END: END,
    },
)

# Compile the graph initially (we’ll recompile with checkpointer + store later)
app = graph.compile(store=memory_store)  # long-term semantic memory for LangMem tools

# Visualize the graph (optional in notebooks)
display(app)


# ================================
# Step 10: Execute the graph (no ST memory)
# ================================
def execute_graph(user_input: str) -> None:
    """
    Stream outputs from the graph

    Args:
        user_input (str): User query string
    """
    for step in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


# Example test runs
execute_graph("What are some best practices for data backups in MongoDB?")
execute_graph("Give me a summary of the page titled Create a MongoDB Deployment")


# ================================
# Step 11: Add short-term memory (checkpointer)
# ================================
# Initialize a MongoDB checkpointer (short-term/thread memory)
checkpointer = MongoDBSaver(
    mongodb_client,
    db_name=DB_NAME,
    collection_name="agent_checkpoints",
)

# Instantiate the graph with the checkpointer AND the long-term store
app = graph.compile(
    checkpointer=checkpointer,  # short-term/thread memory
    store=memory_store,         # long-term semantic memory for LangMem tools
)


def execute_graph_with_memory(thread_id: str, user_input: str) -> None:
    """
    Stream outputs from the graph

    Args:
        thread_id (str): Thread ID for the checkpointer
        user_input (str): User query string
    """
    config = {
        "configurable": {
            "thread_id": thread_id,          # short-term memory key
            "user_id": "devday-demo-user",   # long-term memory namespace key
        }
    }

    for step in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


# Test graph execution with thread ID (short-term + long-term memory)
execute_graph_with_memory(
    "1",
    "What are some best practices for data backups in MongoDB?",
)

# Follow-up question to ensure message history works
execute_graph_with_memory(
    "1",
    "What did I just ask you?",
)


# ================================
# OPTIONAL Step 12: Periodic long-term memory summarizer (cold path)
# ================================
def summarize_long_term_memory(
    user_id: str,
    hours_back: int = 24,
    max_items: int = 50,
) -> str:
    """
    Summarize recent long-term memories for a user and store a compact summary.

    Args:
        user_id (str): User identifier (must match config['configurable']['user_id'])
        hours_back (int): Look-back window for memories to summarize.
        max_items (int): Max number of memories to summarize.

    Returns:
        str: Summary text that was stored as a new memory.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours_back)

    results = memory_store.search(
        namespace=("memories", user_id),
        query="*",
        limit=max_items,
    )

    recent_texts = []
    for item in results:
        created_at = getattr(item, "created_at", None) or item.get("created_at")
        if created_at is None or created_at >= cutoff:
            val = item.get("value", {})
            content = val.get("content") or val.get("text") or str(val)
            recent_texts.append(content)

    if not recent_texts:
        return f"No long-term memories found for last {hours_back} hours."

    summarizer_prompt = LCChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize long-term agent memory. "
                "Keep only stable, important facts or preferences. "
                "Drop ephemeral chatter.",
            ),
            ("human", "Here are memory entries:\n\n{memories}"),
        ]
    )

    summarizer_chain = summarizer_prompt | llm
    summary = summarizer_chain.invoke(
        {"memories": "\n- " + "\n- ".join(recent_texts)}
    ).content

    memory_store.put(
        namespace=("memories", user_id),
        key=f"summary_{datetime.datetime.utcnow().isoformat()}",
        value={
            "content": summary,
            "type": "memory_summary",
            "hours_back": hours_back,
        },
    )

    return summary


def run_memory_maintenance(
    user_id: str = "devday-demo-user",
    hours_back: int = 24,
) -> None:
    """
    Simulate a periodic maintenance job that summarizes long-term memory.
    In production, call this every N hours/days via scheduler.
    """
    print(f"Summarizing long-term memory for user_id={user_id}, hours_back={hours_back} ...")
    summary = summarize_long_term_memory(user_id=user_id, hours_back=hours_back)
    print("\n=== Summary written to long-term memory ===\n")
    print(summary)


# Example usage (run manually in the lab):
# run_memory_maintenance("devday-demo-user", hours_back=24)

