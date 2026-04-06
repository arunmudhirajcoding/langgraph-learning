# langGraph
## Prerequisites
- Python 3.8 or higher
- langchain
- knowlegfe on LLM's

## difference b/w LLM and genAI
### LLM approach
1. generate jd 
2. suggests platforms to post jd
3. suggests peopple to hire based on jd
4. draft email 
5. generate questions 
### problems 
- reactive, not proactive
- No memory
- not for specific data like internally judge by the specific company
- can't perform actions like post jd in platforms 

### rag based chatbot approach
1. provide specific company instructions to feed the llm so that it can generate jd based on company 
2. suggests platforms based on company approaches
3. suggests peopple to hire based on jd
4. draft email 
5. generate questions 
### problems
- reactive, not proactive ❌
- No memory ❌
- specific data like internally judge by the specific company ✔️
- can't perform actions like post jd in platforms ❌

### tools based system
- providing api of tools to operate by llm 
- reactive, not proactive ❌
- No memory ❌
- specific data like internally judge by the specific company ✔️
- perform actions like post jd in platforms✔️
- cant adapt ❌

### AI-Agent 
- proactive ✔️
- memory ✔️
-  specific data like internally judge by the specific company ✔️
- can't perform actions like post jd in platforms✔️
- can adapt ✔️
