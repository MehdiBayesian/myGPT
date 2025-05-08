from typing import Dict, Tuple, Optional, Generator
from dataclasses import dataclass

@dataclass
class ThinkingMarkers:
    start: str
    end: str
    start_message: str = "🤔 [Started Tak-Navazi ...] "
    end_message: str = " [... Done Tak-Navazi] 🏁 "

class StreamBuffer:
    def __init__(self, thinking_markers: Optional[ThinkingMarkers] = None):
        self.buffer = ""
        self.in_thinking_mode = False
        self.thinking_markers = thinking_markers

    def process_chunk(self, content: str) -> Generator[str, None, None]:
        """Process a chunk of content and yield appropriate output."""
        if not self.thinking_markers:
            yield content
            return

        self.buffer += content
        yield from self._process_buffer()

    def _process_buffer(self) -> Generator[str, None, None]:
        """Process the internal buffer and yield appropriate output."""
        while True:
            if not self.in_thinking_mode:
                # Look for start thinking marker
                if self.thinking_markers.start in self.buffer:
                    parts = self.buffer.split(self.thinking_markers.start, 1)
                    if parts[0]:
                        yield parts[0]
                    yield self.thinking_markers.start_message
                    self.buffer = parts[1]
                    self.in_thinking_mode = True
                    continue
            else:
                # Look for end thinking marker
                if self.thinking_markers.end in self.buffer:
                    parts = self.buffer.split(self.thinking_markers.end, 1)
                    if parts[0]:
                        yield parts[0]
                    yield self.thinking_markers.end_message
                    self.buffer = parts[1]
                    self.in_thinking_mode = False
                    continue

            # If no markers found or after processing markers
            if not any(marker in self.buffer for marker in 
                      [self.thinking_markers.start, self.thinking_markers.end]):
                if self.buffer:
                    yield self.buffer
                    self.buffer = ""
            break

    def flush(self) -> Generator[str, None, None]:
        """Flush any remaining content in the buffer."""
        if self.buffer:
            yield self.buffer
            self.buffer = ""

def get_model_thinking_markers(model_name: str, 
                             thinking_markers_config: Dict[str, Tuple[str, str]]) -> Optional[ThinkingMarkers]:
    """Get thinking markers for a given model if available."""
    # Extract base model name (e.g., 'deepseek-r1' from 'deepseek-r1:4b')
    base_model = model_name.split(':')[0].lower() if ':' in model_name else model_name.lower()
    
    if base_model in thinking_markers_config:
        start, end = thinking_markers_config[base_model]
        return ThinkingMarkers(start=start, end=end)
    return None 