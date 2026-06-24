"""Examples demonstrating infy usage."""

from infy import Lambda, Parallel, tool
from infy.agents import create_agent
from infy.parsers import JsonParser

# ---------------------------------------------------------------------------
# Example 1: Basic tool creation
# ---------------------------------------------------------------------------


@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': The weather is sunny today."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # noqa: S307 — demo only
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Example 2: Pipe composition
# ---------------------------------------------------------------------------


def demo_composition():
    """Show pipe operator composition."""
    chain = (
        Lambda(lambda x: x.upper())
        | Lambda(lambda x: f"Processed: {x}")
        | Lambda(lambda x: x + "!")
    )

    result = chain.invoke("hello world")
    print(result)  # "Processed: HELLO WORLD!"


# ---------------------------------------------------------------------------
# Example 3: Parallel execution
# ---------------------------------------------------------------------------


def demo_parallel():
    """Show parallel execution."""
    pipeline = Parallel(
        {
            "upper": Lambda(lambda x: x.upper()),
            "length": Lambda(lambda x: len(x)),
            "reversed": Lambda(lambda x: x[::-1]),
        }
    )

    result = pipeline.invoke("hello")
    print(result)
    # {"upper": "HELLO", "length": 5, "reversed": "olleh"}


# ---------------------------------------------------------------------------
# Example 4: Output parsing
# ---------------------------------------------------------------------------


def demo_parsing():
    """Show output parsing."""
    parser = JsonParser()

    # Parse from markdown code block
    result = parser.parse('```json\n{"name": "infy", "version": 1}\n```')
    print(result)  # {"name": "infy", "version": 1}

    # Parse partial JSON (streaming)
    partials = []
    for chunk in ['{"name":', ' "infy",', ' "version": 1}']:
        partials.append(chunk)
        try:
            result = parser.parse("".join(partials), partial=True)
            if result:
                print(f"Partial: {result}")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Example 5: Agent with OpenAI
# ---------------------------------------------------------------------------


def demo_agent():
    """Show agent creation. Requires OPENAI_API_KEY env var."""
    from infy.providers.openai import OpenAIChat

    model = OpenAIChat("gpt-4o")

    agent = create_agent(
        model,
        tools=[search, calculator],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    result = agent("What is 2 + 2? Also search for 'python langchain alternative'")
    print(f"Response: {result.response.text}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {result.tool_calls_made}")


# ---------------------------------------------------------------------------
# Example 6: Agent with Ollama (local)
# ---------------------------------------------------------------------------


def demo_agent_ollama():
    """Show agent with local Ollama model."""
    from infy.providers.ollama import OllamaChat

    model = OllamaChat("llama3.1")

    agent = create_agent(
        model,
        tools=[search, calculator],
        system_prompt="You are a helpful assistant.",
    )

    result = agent("Calculate 42 * 17")
    print(f"Response: {result.response.text}")


if __name__ == "__main__":
    print("=== Composition ===")
    demo_composition()

    print("\n=== Parallel ===")
    demo_parallel()

    print("\n=== Parsing ===")
    demo_parsing()
