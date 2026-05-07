"""LangGraph ReAct agent for agentic feature evaluation."""

from langchain_openai import ChatOpenAI

# from langgraph.prebuilt import create_react_agent
from src.parser.models import AgentEvaluation
from langchain.agents import create_agent


def build_agent(model: ChatOpenAI, tools: list):
    """
    Build a LangGraph ReAct agent with the specified model and tools.

    Args:
        model: ChatOpenAI model instance
        tools: List of LangGraph tool functions

    Returns:
        Compiled LangGraph agent
    """
    return create_agent(model, tools)


def run_agent(agent, question: str, instructions: str) -> AgentEvaluation:
    """
    Run the agent with a question and instructions, returning structured evaluation.

    Args:
        agent: Compiled LangGraph agent
        question: The evaluation question
        instructions: Additional instructions for the agent

    Returns:
        AgentEvaluation with evaluation (bool) and explanation (str)
    """
    system_message = f"""You are an expert video analyst evaluating YouTube ABCD criteria.

Your task:
1. Query the database using the provided tools to gather relevant information
2. Analyze the results to answer the question
3. Return your evaluation as JSON with two fields:
   - "evaluation": true or false
   - "explanation": a concise explanation (1-3 sentences) of your reasoning

Question: {question}

Instructions:
{instructions}

Use the tools to query the database, then provide your structured evaluation."""

    # Invoke the agent
    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": system_message},
            ]
        }
    )

    # Extract the final message from the agent
    final_message = result["messages"][-1].content

    # Parse the agent's response
    # The agent should return JSON, but we'll handle potential formatting issues
    try:
        import json

        # Try to extract JSON from the response
        if "{" in final_message and "}" in final_message:
            start = final_message.find("{")
            end = final_message.rfind("}") + 1
            json_str = final_message[start:end]
            parsed = json.loads(json_str)

            # Ensure we have the required fields
            evaluation = bool(parsed.get("evaluation", False))
            explanation = str(parsed.get("explanation", "No explanation provided"))

            return AgentEvaluation(evaluation=evaluation, explanation=explanation)
        else:
            # Fallback: try to infer evaluation from response
            evaluation = (
                "true" in final_message.lower() or "yes" in final_message.lower()
            )
            return AgentEvaluation(
                evaluation=evaluation, explanation=final_message[:500]
            )

    except Exception as e:
        # Fallback for parsing errors
        return AgentEvaluation(
            evaluation=False,
            explanation=f"Error parsing agent response: {str(e)}. Raw response: {final_message[:200]}",
        )
