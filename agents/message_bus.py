import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    REQUEST = "request"


class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """A message passed between agents."""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: dict
    priority: Priority = Priority.NORMAL
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    parent_id: str = None   # links replies to original messages
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id
        }


class MessageBus:
    """
    Central message passing system for multi-agent communication.

    Agents do not talk to each other directly — they send messages
    through the bus. The bus routes, queues, and logs all messages.

    Benefits:
    - Decoupled agents — agents don't need to know about each other
    - Message history — full audit trail of all agent communication
    - Priority queuing — urgent tasks handled first
    - Easy to add new agents without changing existing ones
    """

    def __init__(self):
        self.queues: dict[str, list] = {}   # per-agent message queues
        self.message_log: list[Message] = []
        self.message_counter = 0

    def register_agent(self, agent_id: str):
        """Register an agent so it can receive messages."""
        if agent_id not in self.queues:
            self.queues[agent_id] = []
            print(f"[MessageBus] Registered agent: {agent_id}")

    def send(
        self,
        sender: str,
        recipient: str,
        message_type: MessageType,
        content: dict,
        priority: Priority = Priority.NORMAL,
        parent_id: str = None
    ) -> str:
        """Send a message from one agent to another."""
        self.message_counter += 1
        msg_id = f"msg_{self.message_counter:04d}"

        message = Message(
            id=msg_id,
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            priority=priority,
            parent_id=parent_id
        )

        # Add to recipient's queue
        if recipient not in self.queues:
            self.register_agent(recipient)

        # Insert by priority (higher priority = front of queue)
        queue = self.queues[recipient]
        inserted = False
        for i, existing in enumerate(queue):
            if message.priority.value > existing.priority.value:
                queue.insert(i, message)
                inserted = True
                break
        if not inserted:
            queue.append(message)

        self.message_log.append(message)
        print(f"[Bus] {sender} → {recipient}: {message_type.value} | {str(content)[:60]}...")
        return msg_id

    def receive(self, agent_id: str) -> Message | None:
        """Get the next message for an agent. Returns None if queue empty."""
        if agent_id not in self.queues or not self.queues[agent_id]:
            return None
        return self.queues[agent_id].pop(0)

    def receive_all(self, agent_id: str) -> list[Message]:
        """Get all pending messages for an agent."""
        if agent_id not in self.queues:
            return []
        messages = self.queues[agent_id].copy()
        self.queues[agent_id] = []
        return messages

    def broadcast(
        self,
        sender: str,
        message_type: MessageType,
        content: dict
    ):
        """Send a message to all registered agents."""
        for agent_id in self.queues:
            if agent_id != sender:
                self.send(sender, agent_id, message_type, content)

    def get_conversation(self, root_msg_id: str) -> list[Message]:
        """Get all messages in a conversation thread."""
        thread = []
        for msg in self.message_log:
            if msg.id == root_msg_id or msg.parent_id == root_msg_id:
                thread.append(msg)
        return sorted(thread, key=lambda m: m.timestamp)

    def stats(self) -> dict:
        """Return message bus statistics."""
        return {
            "total_messages": len(self.message_log),
            "registered_agents": list(self.queues.keys()),
            "pending_messages": {
                agent: len(queue)
                for agent, queue in self.queues.items()
            }
        }

    def print_log(self, limit: int = 20):
        """Print the message log."""
        print(f"\n{'='*60}")
        print(f"  MESSAGE LOG (last {limit})")
        print(f"{'='*60}")
        for msg in self.message_log[-limit:]:
            print(
                f"  [{msg.timestamp[11:19]}] "
                f"{msg.sender:15} → {msg.recipient:15} | "
                f"{msg.message_type.value:8} | "
                f"{str(msg.content)[:50]}..."
            )