from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
import json


app = FastAPI()

class ConversionRequest(BaseModel):
    choice: str
    api_key: str
    model: str = None

def read_sql_files(files: List[UploadFile]):
    sql_file_list = []
    for file in files:
        sql_content = file.file.read().decode()
        sql_file_list.append(sql_content)
    return sql_file_list

def create_prompt(sql_content):
    return f"""
    ONLY RESPOND WITH A VALID PYTHON CODE. THE CODE IN RESPONSE SHOULD BE IMMEDIATELY RUNNABLE.DO NOT ADD ANY TEXT OTHER THAN THE PYTHON CODE EVER. 
    If there is no code provided below then respond with -> print('Empty'). 
    Make sure to define/initialize any variables that you may use. 
    Make all the necessary imports. 
    Make sure the code is runnable in python version 3.11.9. 
    Your entire response is going to be run by a python compiler. 
    DO NOT ADD python or any other text besides the code. 
    You are tasked with converting .sql file code to .py with PySpark code files. 

    Convert the following SQL file content to PySpark python code:\n\n{sql_content}
    """

def process_with_gpt(sql_contents, api_key):
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model="gpt-4o")
        
        results = []
        for sql_content in sql_contents:
            prompt = create_prompt(sql_content)
            response = llm.invoke(prompt)
            
            code_block = response.content
            if code_block.startswith("```python"):
                code_block = code_block[len("```python"):].strip()
            if code_block.endswith("```"):
                code_block = code_block[:-len("```")].strip()
                
            results.append(code_block)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with GPT: {str(e)}")

def process_with_huggingface(sql_contents, hf_api_key, model):
    try:
        client = InferenceClient(api_key=hf_api_key, model=model)
        results = []
        
        for sql_content in sql_contents:
            prompt = create_prompt(sql_content)
            response = client.post(
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 400},
                    "task": "text-generation",
                }
            )
            
            data = json.loads(response.decode())[0]["generated_text"]
            code_block = data
            if code_block.startswith(prompt):
                code_block = code_block[len(prompt):].strip()
            if code_block.endswith("```"):
                code_block = code_block[:-len("```")].strip()
                
            results.append(code_block)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Hugging Face: {str(e)}")

@app.post("/convert")
async def convert_sql_to_pyspark(
    files: List[UploadFile] = File(...),
    model_choice: str = Form(...),
    api_key: str = Form(...),
    model_name: str = Form(None)
):
    if not files:
        raise HTTPException(status_code=400, detail="No SQL files provided")
    
    sql_contents = read_sql_files(files)
    
    if model_choice == "ChatGPT":
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required")
        results = process_with_gpt(sql_contents, api_key)
    elif model_choice == "Hugging Face Model":
        if not api_key or not model_name:
            raise HTTPException(status_code=400, detail="Hugging Face API key and model name are required")
        results = process_with_huggingface(sql_contents, api_key, model_name)
    else:
        raise HTTPException(status_code=400, detail="Invalid model choice")
    
    conversion_results = []
    for file, pyspark_code in zip(files, results):
        conversion_results.append({
            "filename": file.filename,
            "pyspark_code": pyspark_code
        })
    
    return JSONResponse(content={"results": conversion_results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
