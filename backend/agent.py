from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Annotated
from tools import rag_search_tool, web_search_tool
from schemas import RouteDecision, RagJudge
from llms import LLMModel

memory=MemorySaver()

class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool

class AxonBotAgent:
    def __init__(self):
        llm_models=LLMModel()
        self.router_llm=llm_models.get_router_model()
        self.judge_llm=llm_models.get_judge_model()
        self.answer_llm=llm_models.get_answer_model()

    def router_node(self,state: AgentState,config : RunnableConfig):
        print("\n--- Entering router_node ---")
        query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)

        system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowledge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
        )
    
        if web_search_enabled:
            system_prompt += (
                "You **CAN** use web search for queries that require very current, real-time, or broad general knowledge "
                "that is unlikely to be in a specific, static knowledge base (e.g., today's news, live data, very recent events)."
                "\n\nChoose one of the following routes:"
                "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection (e.g., 'What is X?', 'How does Y work?', 'Explain Z policy')."
                "\n- 'web': For queries about current events, live data, very recent news, or broad general knowledge that requires up-to-date internet access (e.g., 'Who won the election yesterday?', 'What is the weather in London?', 'Latest news on technology')."
            )
        else:
            system_prompt += (
                "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route."
                "If a query would normally require web search, you should attempt to answer it using RAG (if applicable) or directly from your general knowledge."
                "\n\nChoose one of the following routes:"
                "\n- 'rag': For queries about specific entities, historical facts, product details, procedures, or any information that would typically be found in a curated document collection, AND for queries that would normally go to web search but web search is disabled."
            )

        system_prompt += (
            "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
            "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
            "\n\nExample routing decisions:"
            "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
            "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
            "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
            "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
            "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
            "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'"
        )

        messages = [
            ("system", system_prompt),
            ("user", query)
        ]
        result: RouteDecision = self.router_llm.invoke(messages)

        initial_router_decision = result.route 
        router_override_reason = None

        if not web_search_enabled and result.route == "web":
            result.route = "rag" 
            router_override_reason = "Web search disabled by user; redirected to RAG."
            print(f"Router decision overridden: changed from 'web' to 'rag' because web search is disabled.")

        print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")

        out = {
            "route": result.route,
            "web_search_enabled": web_search_enabled
        }

        if router_override_reason: 
            out["initial_router_decision"] = initial_router_decision
            out["router_override_reason"] = router_override_reason

        if result.route == "end":
            out["messages"] = [AIMessage(content=result.reply or "Hello!")]

        return out


    def rag_node(self,state: AgentState,config:RunnableConfig):
        print("\n--- Entering rag_node ---")
        query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)

        print(f"RAG query: {query}")
        chunks = rag_search_tool.invoke(query)

        if chunks.startswith("RAG_ERROR::"):
            print(f"{chunks}. Checking web search enabled status.")
            next_route = "web" if web_search_enabled else "answer"
            return {"rag": "", "route": next_route}
        
        if chunks:
            print(f"Retrieved RAG chunks (first 500 chars): {chunks[:500]}...")
        else:
            print("No RAG chunks retrieved.")

        judge_messages = [
            ("system", (
                "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
                "to fully and accurately answer the user's question. "
                "Consider if the retrieved text directly addresses the question's core and provides enough detail."
                "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
                "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
                "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient."
                "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
                "\n\nExample 1: Question: 'What is the capital of France?' Retrieved: 'Paris is the capital of France.' -> {\"sufficient\": true}"
                "\nExample 2: Question: 'What are the symptoms of diabetes?' Retrieved: 'Diabetes is a chronic condition.' -> {\"sufficient\": false} (Doesn't answer symptoms)"
                "\nExample 3: Question: 'How to fix error X in software Y?' Retrieved: 'No relevant information found.' -> {\"sufficient\": false}"
            )),
            ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
        ]

        verdict: RagJudge = self.judge_llm.invoke(judge_messages)
        print(f"RAG Judge verdict: {verdict.sufficient}")
        print("--- Exiting rag_node ---")

        if verdict.sufficient:
            next_route = "answer"
        else:
            next_route = "web" if web_search_enabled else "answer" 
            print(f"RAG not sufficient. Web search enabled: {web_search_enabled}. Next route: {next_route}")
        
        return {
            "rag": chunks,
            "route": next_route,
            "web_search_enabled": web_search_enabled 
        }
        
        

    def web_node(self,state: AgentState,config:RunnableConfig):
        print("\n--- Entering web_node ---")
        query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)

        if not web_search_enabled:
            print("Web search node entered but web search is disabled. Skipping actual search.")
            return {"web": "Web search was disabled by the user.", "route": "answer"}
        
        print(f"Web search query: {query}")
        snippets = web_search_tool.invoke(query)

        if snippets.startswith("WEB_ERROR::"):
            print(f"{snippets}. Proceeding to answer with limited info.")
            return {"web": "", "route": "answer"}
        
        print(f"Web snippets retrieved: {snippets[:200]}...")
        print("--- Exiting web_node ---")
        return {"web": snippets, "route": "answer"}
    

    def answer_node(self,state: AgentState):
        print("\n--- Entering answer_node ---")
        user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

        ctx_parts = []
        if state.get("rag"):
            ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
        if state.get("web") and not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])

        context = "\n\n".join(ctx_parts)

        if not context.strip():
            context = "No external context was available for this query. Try to answer based on general knowledge if possible."

        prompt = f"""Please answer the user's question using the provided context.
                If the context is empty or irrelevant, try to answer based on your general knowledge.

                Question: {user_q}

                Context:
                {context}

                Provide a helpful, accurate, and concise response based on the available information."""
        
        print(f"Prompt sent to answer_llm: {prompt[:500]}...")
        ans = self.answer_llm.invoke([HumanMessage(content=prompt)]).content
        print(f"Final answer generated: {ans[:200]}...")
        print("--- Exiting answer_node ---")
        return {"messages": [AIMessage(content=ans)]}
    
    def from_router(self,st: AgentState) -> Literal["rag", "web", "answer", "end"]:
        return st["route"]
    
    def after_rag(self,st: AgentState) -> Literal["answer", "web"]:
        return st["route"]

    def workflow(self):
        self.graph=StateGraph(AgentState)
        self.graph.add_node("router",self.router_node)
        self.graph.add_node("rag_lookup",self.rag_node)
        self.graph.add_node("web_search",self.web_node)
        self.graph.add_node("answer",self.answer_node)

        self.graph.set_entry_point("router")

        self.graph.add_conditional_edges("router",self.from_router,{
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END
        })        
        self.graph.add_conditional_edges("rag_lookup",self.after_rag,{
            "web": "web_search",
            "answer": "answer"
        })

        self.graph.add_edge("web_search","answer")
        self.graph.add_edge("answer",END)

        self.app=self.graph.compile(checkpointer=memory)
        return self.app