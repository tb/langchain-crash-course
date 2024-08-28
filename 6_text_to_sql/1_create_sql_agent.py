from dotenv import load_dotenv
from time import monotonic

from langchain_community.utilities import  SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()

db = SQLDatabase.from_uri(f"postgresql://localhost:5432/northwind")
context = db.get_context()

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

example_query = "show me gross sales in Europe vs USA"

start_time = monotonic()

agent_executor.invoke({"input": example_query})

print(f"Run time {monotonic() - start_time} seconds")