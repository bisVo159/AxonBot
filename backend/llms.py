from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import RouteDecision, RagJudge
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LLMModel:
    def __init__(self,
                 router_model_name="gemini-2.5-flash",
                 judge_model_name='llama-3.3-70b-versatile',
                 answer_model_name='gemini-2.5-flash'
                 ):
        self.router_model= ChatGoogleGenerativeAI(model=router_model_name,temperature=0.1).with_structured_output(RouteDecision)
        self.judge_model= ChatGroq(model=judge_model_name,temperature=0).with_structured_output(RagJudge)
        self.answer_model= ChatGoogleGenerativeAI(model=answer_model_name,temperature=0.5)

    def get_router_model(self):
        return self.router_model 
        
    def get_judge_model(self):
        return self.judge_model
    
    def get_answer_model(self):
        return self.answer_model