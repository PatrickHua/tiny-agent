# TinyAgent

A simple AI agent implementation that can execute tools and maintain conversation history.

## Features

- **Tool Execution**: Execute various tools with parameters
- **Conversation History**: Maintain a history of user messages and agent responses
- **Flexible Tool Integration**: Easy to extend with custom tools

## Usage

```python
from tiny_agent import TinyAgent

# Create an agent instance
agent = TinyAgent()

# Add a message to the conversation
agent.add_message("user", "Hello, can you help me with something?")

# Execute a tool (example)
result = agent.execute_tool("example_tool", {"param1": "value1"})

# Get conversation history
history = agent.get_history()
```

## Class Overview

### TinyAgent

The main agent class with the following methods:

- `add_message(role, content)`: Add a message to the conversation history
  - `role`: The role of the message sender (e.g., "user", "assistant")
  - `content`: The message content

- `execute_tool(tool_name, parameters)`: Execute a specified tool with given parameters
  - `tool_name`: Name of the tool to execute
  - `parameters`: Dictionary of parameters for the tool

- `get_history()`: Returns the complete conversation history

## Installation

Simply include the `tiny_agent.py` file in your project and import the `TinyAgent` class.

## Requirements

- Python 3.6+

## Example

```python
# Initialize the agent
agent = TinyAgent()

# Start a conversation
agent.add_message("user", "What's the weather like?")
agent.add_message("assistant", "I'd be happy to help you check the weather. Let me use the weather tool.")

# Execute a weather tool (if implemented)
weather_result = agent.execute_tool("weather", {"location": "New York"})

# View conversation history
for message in agent.get_history():
    print(f"{message['role']}: {message['content']}")
```

## Contributing

Feel free to contribute by adding new tools or improving the agent's functionality.

## License

This project is open source and available under the MIT License.