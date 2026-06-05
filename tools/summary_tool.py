from config import LLM

def summary_tool(user_query, context):
    """
    Summarizes the provided context based on specific user instructions.
    
    Args:
        user_query (str): The user's specific request (e.g., "summarize in 1 para").
        context (str): The text extracted from the PDF.
    """
    
    # We use a structured prompt to ensure the LLM prioritizes the 'Task' constraints
    prompt = f"""
    SYSTEM ROLE: You are an expert Document Analyst.
    
    TASK: {user_query}
    
    PRIMARY RULE: You must follow the formatting, length, and style requested in the TASK above. 
    If the user asks for "1 paragraph," do not provide more.
    
    SOURCE MATERIAL:
    {context}
    
    RESPONSE:
    """
    
    # Using .stream for the UI experience in Streamlit
    return LLM.stream(prompt)