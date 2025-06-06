# LangGraph Multi-Agent Example

This project demonstrates how to build both single-agent and multi-agent systems using [LangGraph](https://github.com/langchain-ai/langgraph), a framework for orchestrating stateful LLM workflows.

## 🔍 What You'll Learn

- How to create a simple ReAct agent using Google Gemini and Serper Search
- How to build a conditional router that sends queries to either:
  - a search agent
  - a math agent
- How to use LangGraph to manage state and agent routing
- How to use the LLM itself to determine routing logic

## 🚀 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

- `main.py`: Contains the full LangGraph implementation with routing and agents
- `requirements.txt`: Required packages
- `README.md`: This guide

## 🧠 How It Works

1. User inputs a query.
2. A router agent uses the LLM to decide whether the query is a math problem or general knowledge.
3. The appropriate agent handles the request.
4. LangGraph manages the state and transitions between nodes.

## 🔗 Inspired By

This project is based on the Medium article:  
**“Building Multi-Agent Systems with LangGraph: A Step-by-Step Guide”**

Happy experimenting! 🔧
