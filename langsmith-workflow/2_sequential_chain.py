from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGSMITH_PROJECT'] = 'langsmith-sequential-chain'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model='moonshotai/kimi-k2-instruct-0905')
model2 = ChatGroq(model='openai/gpt-oss-120b',temperature=0.7)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# user defined tags and metadata
config = {
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {'model1':' moonshotai/kimi-k2-instruct-0905', 'model2':'openai/gpt-oss-120b'}
}
result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
