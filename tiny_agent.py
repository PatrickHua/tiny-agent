#!/usr/bin/env python3
"""
Cline Core - Minimal Python Implementation
A simplified coding agent with conversation loop, tool execution, and streaming.

Based on the original Cline architecture: https://github.com/cline/cline

Setup: pip install anthropic && export ANTHROPIC_API_KEY="your-key"
Usage: python cline_core.py
"""

import asyncio
import json
import re
import time
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, AsyncGenerator
from abc import ABC, abstractmethod

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Install anthropic: pip install anthropic")


@dataclass
class TextBlock:
    type: str = "text"
    content: str = ""
    partial: bool = False


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    params: Dict = field(default_factory=dict)
    partial: bool = False


@dataclass
class StreamChunk:
    type: str
    text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class TaskState:
    is_streaming: bool = False
    current_streaming_content_index: int = 0
    assistant_message_content: List[Union[TextBlock, ToolUseBlock]] = field(default_factory=list)
    user_message_content: List[Dict] = field(default_factory=list)
    user_message_content_ready: bool = False
    abort: bool = False


class MessageParser:
    @staticmethod
    def parse_assistant_message(message: str) -> List[Union[TextBlock, ToolUseBlock]]:
        content_blocks = []
        tool_pattern = r'<(read_file|write_to_file|execute_command|list_files|attempt_completion)>(.*?)</\1>'
        last_end = 0
        
        for match in re.finditer(tool_pattern, message, re.DOTALL):
            start, end = match.span()
            
            if start > last_end:
                text_content = message[last_end:start].strip()
                if text_content:
                    content_blocks.append(TextBlock(content=text_content))
            
            tool_name = match.group(1)
            tool_content = match.group(2).strip()
            
            params = {}
            param_pattern = r'<(\w+)>(.*?)</\1>'
            for param_match in re.finditer(param_pattern, tool_content, re.DOTALL):
                params[param_match.group(1)] = param_match.group(2).strip()

            content_blocks.append(ToolUseBlock(name=tool_name, params=params))
            last_end = end
        
        if last_end < len(message):
            remaining_text = message[last_end:].strip()
            if remaining_text:
                content_blocks.append(TextBlock(content=remaining_text))
        
        if not content_blocks and message.strip():
            content_blocks.append(TextBlock(content=message.strip()))
        
        return content_blocks


class APIProvider(ABC):
    @abstractmethod
    async def create_message_stream(self, system_prompt: str, messages: List[Dict]) -> AsyncGenerator[StreamChunk, None]:
        pass


class AnthropicAPIProvider(APIProvider):
    def __init__(self):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Run: pip install anthropic")
        
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Set ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    async def create_message_stream(self, system_prompt: str, messages: List[Dict]) -> AsyncGenerator[StreamChunk, None]:
        anthropic_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        
        with self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                yield StreamChunk(type="text", text=text)
            
            message = stream.get_final_message()
            yield StreamChunk(
                type="usage",
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens
            )


class ToolExecutor:
    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd)
    
    async def execute_tool(self, tool: ToolUseBlock) -> str:
        try:
            if tool.name == "read_file":
                return await self._read_file(tool.params.get("path", ""))
            elif tool.name == "write_to_file":
                return await self._write_file(tool.params.get("path", ""), tool.params.get("content", ""))
            elif tool.name == "execute_command":
                return await self._execute_command(tool.params.get("command", ""))
            elif tool.name == "list_files":
                return await self._list_files(tool.params.get("path", "."))
            elif tool.name == "attempt_completion":
                return f"Task completed: {tool.params.get('result', 'Done')}"
            else:
                return f"Unknown tool: {tool.name}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _read_file(self, path: str) -> str:
        file_path = self.cwd / path
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File contents of {path}:\n```\n{content}\n```"
    
    async def _write_file(self, path: str, content: str) -> str:
        file_path = self.cwd / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"
    
    async def _execute_command(self, command: str) -> str:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd
        )
        stdout, stderr = await process.communicate()
        
        result = []
        if stdout:
            result.append(f"STDOUT:\n{stdout.decode()}")
        if stderr:
            result.append(f"STDERR:\n{stderr.decode()}")
        result.append(f"Exit code: {process.returncode}")
        return "\n".join(result)
    
    async def _list_files(self, path: str) -> str:
        dir_path = self.cwd / path
        if not dir_path.exists():
            return f"Directory {path} does not exist"
        
        files = []
        for item in sorted(dir_path.iterdir()):
            if item.is_dir():
                files.append(f"📁 {item.name}/")
            else:
                files.append(f"📄 {item.name}")
        return f"Contents of {path}:\n" + "\n".join(files)


class Task:
    def __init__(self, api_provider: APIProvider, cwd: str = "."):
        self.api_provider = api_provider
        self.task_state = TaskState()
        self.tool_executor = ToolExecutor(cwd)
        self.conversation_history: List[Dict] = []
        
        self.system_prompt = """You are Cline, an AI coding assistant.

Available tools:
- read_file: Read file contents
- write_to_file: Create or modify files
- execute_command: Run shell commands
- list_files: List directory contents  
- attempt_completion: Mark task as complete

Tool format examples:
<read_file>
<path>filename.py</path>
</read_file>

<write_to_file>
<path>filename.py</path>
<content>file content here</content>
</write_to_file>

<execute_command>
<command>ls -la</command>
</execute_command>

Use tools systematically to complete tasks. Call attempt_completion when finished."""

    async def start_task(self, initial_task: str) -> None:
        self.conversation_history = [{"role": "user", "content": initial_task}]
        await self._task_loop(initial_task)
    
    async def _task_loop(self, initial_content: str) -> None:
        user_content = [{"type": "text", "text": initial_content}]
        
        while not self.task_state.abort:
            did_end = await self._make_request(user_content)
            if did_end:
                # Check if we have a next task to continue with
                if hasattr(self, 'next_task') and self.next_task:
                    next_task_content = self.next_task
                    self.next_task = None  # Clear it
                    
                    # Reset state and continue with next task
                    self.task_state.abort = False
                    user_content = [{"type": "text", "text": next_task_content}]
                    continue
                else:
                    break
            else:
                user_content = [{"type": "text", "text": "Continue with the task or call attempt_completion."}]
    
    async def _make_request(self, user_content: List[Dict]) -> bool:
        if self.task_state.abort:
            return True
        
        # Reset state
        self.task_state.current_streaming_content_index = 0
        self.task_state.assistant_message_content = []
        self.task_state.user_message_content = []
        self.task_state.user_message_content_ready = False
        
        # Prepare messages
        messages = self.conversation_history.copy()
        content_text = "\n".join(item.get("text", "") for item in user_content if item.get("text"))
        messages.append({"role": "user", "content": content_text})
        
        print("\n🤖 Thinking...")
        
        # Stream response
        assistant_message = ""
        self.task_state.is_streaming = True
        
        try:
            async for chunk in self.api_provider.create_message_stream(self.system_prompt, messages):
                if chunk.type == "text" and chunk.text:
                    assistant_message += chunk.text
                    print(chunk.text, end="", flush=True)
                    
                    self.task_state.assistant_message_content = MessageParser.parse_assistant_message(assistant_message)
                    await self._present_content()
                    
                    if self.task_state.abort:
                        break
                
                elif chunk.type == "usage":
                    print(f"\n📊 {chunk.input_tokens} in, {chunk.output_tokens} out")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return True
        
        finally:
            self.task_state.is_streaming = False
        
        print()
        
        if self.task_state.abort:
            return True
        
        # Finalize content
        for block in self.task_state.assistant_message_content:
            block.partial = False
        await self._present_content()
        
        while not self.task_state.user_message_content_ready and not self.task_state.abort:
            await asyncio.sleep(0.1)
        
        if self.task_state.abort:
            return True
        
        # Add to history
        if assistant_message:
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            tool_blocks = [b for b in self.task_state.assistant_message_content if isinstance(b, ToolUseBlock)]
            if not tool_blocks:
                return False
        
        return True
    
    async def _present_content(self) -> None:
        if self.task_state.current_streaming_content_index >= len(self.task_state.assistant_message_content):
            if not self.task_state.is_streaming:
                self.task_state.user_message_content_ready = True
            return
        
        block = self.task_state.assistant_message_content[self.task_state.current_streaming_content_index]
        
        if isinstance(block, ToolUseBlock) and not block.partial:
            print(f"\n🔧 {block.name}")
            
            if block.name == "attempt_completion":
                result = block.params.get('result', 'Task completed')
                print(f"✅ {result}")
                
                next_task = input("\nNext task (or 'quit' to exit): ").strip()
                if next_task.lower() in ['quit', 'exit', 'q']:
                    self.task_state.abort = True
                elif next_task:
                    # Store the next task for the main loop to handle
                    self.next_task = next_task
                    self.task_state.abort = True  # Exit current conversation but continue to next task
                else:
                    # No input given, treat as task completion
                    self.task_state.abort = True
                
                self.task_state.user_message_content_ready = True
                return
            
            result = await self.tool_executor.execute_tool(block)
            print(f"📋 {result}")
            
            self.task_state.user_message_content.append({
                "type": "text",
                "text": f"Tool {block.name} result:\n{result}"
            })
        
        if not block.partial:
            if self.task_state.current_streaming_content_index == len(self.task_state.assistant_message_content) - 1:
                self.task_state.user_message_content_ready = True
            
            self.task_state.current_streaming_content_index += 1
            
            if self.task_state.current_streaming_content_index < len(self.task_state.assistant_message_content):
                await self._present_content()


async def main():
    print("🤖 Cline Core - Minimal")
    
    try:
        api_provider = AnthropicAPIProvider()
    except Exception as e:
        print(f"❌ {e}")
        return
    
    while True:
        task_input = input("\n🎯 Task (or 'quit'): ").strip()
        if not task_input or task_input.lower() in ['quit', 'exit']:
            break
        
        task = Task(api_provider, cwd=".")
        try:
            await task.start_task(task_input)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!") 