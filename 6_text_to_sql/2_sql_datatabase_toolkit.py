from dotenv import load_dotenv
from time import monotonic

from langchain_community.utilities import  SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent

load_dotenv()

db = SQLDatabase.from_uri(f"postgresql://localhost:5432/northwind")
context = db.get_context()

# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# https://smith.langchain.com/hub/langchain-ai/sql-agent-system-prompt
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
# print(prompt_template.input_variables)

system_message = prompt_template.format(dialect=db.dialect, top_k=5)

agent = create_react_agent(llm, toolkit.get_tools(), state_modifier=system_message)

example_query = "show me gross sales in Europe vs USA"

start_time = monotonic()

result = agent.invoke({"messages": [("user", example_query)]})

print(result['messages'][-1])

print(f"Run time {monotonic() - start_time} seconds")

# NOTE:
# langchain_community.agent_toolkits.sql.toolkit SQLDatabaseToolkit
# is about 30% faster than langchain_community.agent_toolkits create_sql_agent