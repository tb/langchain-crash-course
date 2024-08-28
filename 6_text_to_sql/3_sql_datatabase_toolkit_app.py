from dotenv import load_dotenv
import chainlit as cl

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

system_message = prompt_template.format(dialect=db.dialect, top_k=5)

@cl.on_message
async def main(message: str):
    agent = create_react_agent(llm, toolkit.get_tools(), state_modifier=system_message)

    result = agent.invoke({"messages": [("user", message.content)]})
    response = result['messages'][-1]

    await cl.Message(content=response.content).send()

# Run: 
# chainlit run 6_text_to_sql/3_sql_datatabase_toolkit_app.py -w
# 
# show me gross sales in Europe vs USA