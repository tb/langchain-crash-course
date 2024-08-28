from dotenv import load_dotenv
from time import monotonic

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent

load_dotenv()

db = SQLDatabase.from_uri(f"postgresql://localhost:5432/northwind")
context = db.get_context()



# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# https://smith.langchain.com/hub/langchain-ai/sql-agent-system-prompt
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

system_message = prompt_template.format(dialect=db.dialect, top_k=5)

agent_executor = create_react_agent(
    llm, toolkit.get_tools(), state_modifier=system_message
)

# example_query = "show orders avg value by country"
# example_query = "show me all employees data"
# example_query = "show me all employees data in a table"
# example_query = "which country's customers spent the most?"
example_query = "show me gross sales in Europe vs USA"
# example_query = "can you show me a summary of my sales for the past week?"

start_time = monotonic()

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()
    # print(event["messages"]) # more detailed

print(f"Run time {monotonic() - start_time} seconds")
