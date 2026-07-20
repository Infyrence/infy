"""Quickstart: a governed agent wired to your Infyrence dashboard with one key.

    export INFYRENCE_API_KEY=sk-...     # from your Infyrence dashboard
    pip install "infy[openai]"
    python examples/infyrence_quickstart.py

Every tool decision streams to your dashboard in real time, and the destructive call
is blocked by policy before it can run. Open the dashboard and watch the feed go Live.
"""

from infy import create_agent, tool
from infy.governance import Policy
from infy.infyrence import connect


@tool(verb="READ")
def get_logs(service: str) -> str:
    """Read recent logs for a service."""
    return f"{service}: error rate 12%, orders table timing out"


@tool(verb="DB", side_effect=True, risk_tier="critical")
def delete_database(name: str) -> str:
    """Delete a database. Destructive and irreversible."""
    return f"deleted {name}"  # never runs: the policy denies it


def main() -> None:
    model, gov = connect(
        agent="sre-responder",
        policy=Policy(deny=["delete_database"], require_approval=["deploy"]),
        model="gpt-4o",
    )
    agent = create_agent(
        model,
        [get_logs, delete_database],
        governance=gov,
        parallel_tools=False,
    )
    result = agent("Investigate the checkout outage, then clean up the orders database.")

    print(result.response.content)
    print(f"\nAudit chain intact: {gov.audit.verify()}")
    print("Open your Infyrence dashboard to see the decisions stream in.")


if __name__ == "__main__":
    main()
