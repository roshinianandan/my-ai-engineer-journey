import argparse
from agents.message_bus import MessageBus
from agents.orchestrator import Orchestrator


def run_research_task(topic: str, use_critic: bool = True) -> dict:
    """
    Run a full research task through the multi-agent system.
    """
    bus = MessageBus()
    orchestrator = Orchestrator(bus)
    result = orchestrator.run(topic, use_critic=use_critic)

    print(f"\n{'='*60}")
    print("  MESSAGE BUS LOG")
    print(f"{'='*60}")
    bus.print_log()
    bus_stats = bus.stats()
    print(f"\n  Total messages: {bus_stats['total_messages']}")
    print(f"  Agents: {bus_stats['registered_agents']}")

    return result


def demo():
    """Run demo research tasks."""
    tasks = [
        "Research RAG systems and explain how they work and why they are useful",
        "Explain the difference between supervised and unsupervised learning with examples",
    ]

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task}")
        print(f"{'#'*60}")
        result = run_research_task(task, use_critic=True)
        print(f"\n{'#'*60}\n")


def interactive():
    """Interactive multi-agent research session."""
    bus = MessageBus()
    orchestrator = Orchestrator(bus)

    print(f"\n🤖 Multi-Agent Research System")
    print(f"   Agents: Searcher, Summarizer, Critic, Writer")
    print(f"   Type any research question to activate the system.")
    print(f"   Type 'bus' to see message log | 'quit' to exit\n")
    print("-" * 60)

    while True:
        try:
            task = input("\nResearch Task: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not task:
            continue

        if task.lower() == "quit":
            break

        if task.lower() == "bus":
            bus.print_log()
            print(f"\nStats: {bus.stats()}")
            continue

        result = orchestrator.run(task, use_critic=True)

        print(f"\n[Done in {result['time_seconds']}s | "
              f"{result['subtasks_completed']} subtasks]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Research System")
    parser.add_argument("--task",  type=str, help="Research task to execute")
    parser.add_argument("--demo",  action="store_true", help="Run demo tasks")
    parser.add_argument("--chat",  action="store_true", help="Interactive mode")
    parser.add_argument("--no-critic", action="store_true", help="Skip critic review")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.task:
        run_research_task(args.task, use_critic=not args.no_critic)
    elif args.chat:
        interactive()
    else:
        interactive()