from dotenv import load_dotenv
from time import monotonic
import chainlit as cl

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

@cl.on_message
async def main(message: cl.Message):
    stream = agent_executor.stream(
        {"messages": [("user", message.content)]},
        stream_mode="values",
    )

    msg = await cl.Message(content="").send()

    for event in stream:
        last_message = event["messages"][-1]
        if last_message.type == "tool":
            # pprint.pp(last_message)
            await msg.stream_token(last_message.name + "\n")
        if last_message.type == "ai":
            # pprint.pp(last_message)
            total_tokens = last_message.response_metadata["token_usage"]["total_tokens"]
            if last_message.content:
                await msg.stream_token("\n\n" + last_message.content + "\n\n &rarr;"+ str(total_tokens) + " tokens final answer\n")
            else:
                await msg.stream_token("&rarr; "+ str(total_tokens) + " tokens processing ")

    await msg.update()

# Run: 
# chainlit run 6_text_to_sql/5_sql_toolkit_stream_app.py -w
# 
# show me gross sales in Europe vs USA