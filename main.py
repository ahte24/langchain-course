from dotenv import load_dotenv

load_dotenv()
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
)

chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input": "Search for 3 jobs for an ai engineer using langchain in banglore or hyderabad area and list there details with apply link"
        },
    )
    print(result)


if __name__ == "__main__":
    main()

