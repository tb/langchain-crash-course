from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

# Setup Postgres chat history https://github.com/langchain-ai/langchain-postgres
sync_connection = psycopg.connect("postgresql://localhost/chat_history")
table_name = "chat_messages"

# Create the table schema in the database and create relevant indexes.
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

session_id = "b7ec8322-aa25-4bc2-be10-b9ade2d4dab1" # Must be a valid str(uuid.uuid4())

chat_history = PostgresChatMessageHistory(
    table_name,
    session_id,
    sync_connection=sync_connection
)

print("Chat History for session " + session_id)
print(chat_history)

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.add_message(HumanMessage(content=query))
    result = model.invoke(query)
    response = result.content
    chat_history.add_message(AIMessage(content=response))
    print(f"AI: {response}")
