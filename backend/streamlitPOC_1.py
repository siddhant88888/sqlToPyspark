import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
import json
# import torch

def read_sql_files(uploaded_files):
    sql_file_list = []
    for uploaded_file in uploaded_files:
        sql_content = uploaded_file.getvalue().decode()
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
        st.error(f"Error processing with GPT: {str(e)}")
        return []

def process_with_huggingface(sql_contents, hf_api_key, model_name):
    try:
        client = InferenceClient(api_key=hf_api_key, model=model_name)
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
        st.error(f"Error processing with Hugging Face: {str(e)}")
        return []

def main():
    st.title("SQL to PySpark Converter")
    
    # File upload
    st.header("Upload SQL Files")
    uploaded_files = st.file_uploader("Choose SQL files", accept_multiple_files=True, type=['sql'])
    
    # Model selection
    model_choice = st.radio(
        "Select Model",
        ["ChatGPT", "Hugging Face Model"]
    )
    
    # API key input
    if model_choice == "ChatGPT":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
    else:
        hf_api_key = st.text_input("Enter Hugging Face API Key", type="password")
        model_name = st.text_input("Enter Hugging Face Model Name")
    
    if st.button("Convert"):
        if not uploaded_files:
            st.warning("Please upload at least one SQL file")
            return
            
        sql_contents = read_sql_files(uploaded_files)
        
        with st.spinner("Converting SQL to PySpark..."):
            if model_choice == "ChatGPT":
                if not api_key:
                    st.warning("Please enter your OpenAI API key")
                    return
                results = process_with_gpt(sql_contents, api_key)
            else:
                if not hf_api_key or not model_name:
                    st.warning("Please enter both Hugging Face API key and model name")
                    return
                results = process_with_huggingface(sql_contents, hf_api_key, model_name)
        
        # Display results
        if results:
            st.success("Conversion completed!")
            
            # Create tabs for each file
            if len(results) > 1:
                tabs = st.tabs([f"File {i+1}: {file.name}" for i, file in enumerate(uploaded_files)])
                
                for i, (tab, sql_file, pyspark_code) in enumerate(zip(tabs, uploaded_files, results)):
                    with tab:
                        st.subheader(f"Conversion Result for {sql_file.name}")
                        st.code(pyspark_code, language="python")
                        
                        # Download button with unique key for each file
                        st.download_button(
                            label=f"Download PySpark code for {sql_file.name}",
                            data=pyspark_code,
                            file_name=f"pyspark_{sql_file.name.replace('.sql', '.py')}",
                            mime="text/plain",
                            key=f"download_btn_{i}"  # Add unique key here
                        )
            else:
                # Single file case
                sql_file = uploaded_files[0]
                pyspark_code = results[0]
                st.subheader(f"Conversion Result for {sql_file.name}")
                st.code(pyspark_code, language="python")
                
                st.download_button(
                    label=f"Download PySpark code for {sql_file.name}",
                    data=pyspark_code,
                    file_name=f"pyspark_{sql_file.name.replace('.sql', '.py')}",
                    mime="text/plain",
                    key="download_btn_single"  # Unique key for single file case
                )

if __name__ == "__main__":
    main()