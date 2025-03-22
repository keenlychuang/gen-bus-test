"""Prompt loader utility"""

import yaml
from pathlib import Path
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

def load_prompt(prompt_name, directory=None):
    """Load prompt from YAML file"""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    prompts_dir = project_root / "prompts"
    
    if directory:
        prompts_dir = prompts_dir / directory
        
    prompt_path = prompts_dir / f"{prompt_name}.yaml"
    
    with open(prompt_path, "r") as f:
        data = yaml.safe_load(f)
    
    if data.get("type") == "chat":
        messages = []
        for message in data.get("messages", []):
            if message["role"] == "placeholder":
                messages.append(MessagesPlaceholder(variable_name=message["variable_name"]))
            else:
                messages.append((message["role"], message["content"]))
        return ChatPromptTemplate.from_messages(messages)
    else:
        return PromptTemplate.from_template(data["template"])