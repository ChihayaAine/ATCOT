# ATCOT: Adaptive Tool-Augmented Chain of Thought

ATCOT is a dynamic iterative reasoning framework that enables Large Language Models (LLMs) to revise their reasoning trajectories based on tool feedback. Unlike traditional forward-only approaches like ReAct, ATCOT provides adaptive correction mechanisms that allow models to revisit and revise previous steps when tools provide contradicting or clarifying information.

## Features

- **Adaptive Planning**: Dynamic plan generation with confidence scoring and dependency tracking
- **Tool-Augmented Execution**: Sophisticated tool selection and parallel candidate generation
- **Correction Mechanism**: Backward traversal for minimal revision set identification
- **State Tracking**: Comprehensive state representation with planning, reasoning, tool history, and corrections
- **Convergence Guarantees**: Bounded corrections with monotonic improvement tracking
- **Flexible Tool System**: Pluggable tool architecture with built-in tools for math, search, code execution

## Architecture

The ATCOT framework implements a comprehensive state representation `S = {P, R, H, C}` where:

- **P**: Planning structure with ordered sequence of steps and explicit dependencies  
- **R**: Reasoning trace comprising intermediate conclusions and justifications
- **H**: Tool invocation history with temporal annotations and results
- **C**: Correction log tracking all revisions and triggering conditions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ATCOT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (optional, for real LLM providers):
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Quick Start

### Basic Usage

```python
import asyncio
from atcot import ATCOTFramework
from atcot.utils.config import load_config

# Load configuration
config = load_config()

# Setup framework
framework = setup_atcot_framework()

# Run a query
async def main():
    result = await framework.execute("Calculate the compound interest on $1000 at 5% for 3 years")
    print(f"Answer: {result.final_answer}")
    print(f"Corrections: {result.total_corrections}")

asyncio.run(main())
```

### Command Line Interface

```bash
# Run a single query
python main.py --query "What is the population of Tokyo in 2023?"

# Interactive mode
python main.py --interactive

# Debug mode
python main.py --query "Calculate 15% of 250" --debug

# Use custom configuration
python main.py --config config.json --query "Your question here"
```

### Configuration

Create a `config.json` file to customize the framework:

```json
{
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4",
    "temperature": 0.7
  },
  "correction": {
    "max_corrections": 5,
    "contradiction_threshold": 0.7
  },
  "tools": {
    "calculator": {"enabled": true},
    "web_search": {"enabled": true, "config_params": {"search_engine": "duckduckgo"}},
    "python_interpreter": {"enabled": true},
    "wikipedia": {"enabled": true, "config_params": {"language": "en"}}
  }
}
```

## Built-in Tools

- **Calculator**: Mathematical computations and expression evaluation
- **Web Search**: Real-time information retrieval from the web
- **Python Interpreter**: Code execution for complex calculations and data processing
- **Wikipedia**: Factual information retrieval from Wikipedia

## Framework Components

### State Representation
- `ATCOTState`: Main state container with P, R, H, C components
- `PlanningStructure`: DAG-based plan representation with dependencies
- `ReasoningTrace`: Sequential reasoning steps with justifications
- `ToolHistory`: Complete tool invocation history
- `CorrectionLog`: Revision tracking and analysis

### Planning System
- `AdaptivePlanner`: Handles initial planning and dynamic replanning
- `LLMPlanGenerator`: LLM-based plan generation with MAP estimation
- Confidence scoring and dependency validation

### Execution Engine
- `ToolAugmentedExecutor`: Main execution coordinator
- `LLMToolSelector`: Intelligent tool selection using learned policies
- Candidate observation generation and reliability scoring

### Correction Mechanism
- `AdaptiveCorrectionMechanism`: Main correction coordinator
- `BackwardTraversalRevision`: Minimal revision set identification
- Convergence tracking and loop prevention

## Advanced Usage

### Custom Tools

```python
from atcot.tools.base import BaseTool, ToolResult

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="Your custom tool description",
            capabilities=["custom_capability"]
        )
    
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        # Your tool implementation
        return ToolResult(content="result", success=True)
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"]
        }

# Register with framework
tool_registry.register_tool(CustomTool())
```

### Custom LLM Interface

```python
from atcot.utils.llm_interface import LLMInterface

class CustomLLMInterface(LLMInterface):
    async def generate_async(self, prompt: str, **kwargs) -> str:
        # Your LLM implementation
        return "Generated response"
```

## Methodology

ATCOT implements the algorithm described in our paper:

1. **Initialize State**: Create comprehensive state representation
2. **Generate Plan**: Decompose query into structured plan with dependencies
3. **Execute Steps**: For each ready step:
   - Generate candidate observations through tool execution
   - Select best observation using reliability scoring
   - Check for local contradictions
   - Perform correction if needed
4. **Global Consistency**: Verify global consistency and replan if necessary
5. **Convergence**: Check for convergence through bounded corrections
6. **Generate Answer**: Synthesize final answer from reasoning trace

## Performance

ATCOT demonstrates consistent improvements over baseline methods:

- **GSM8K**: 1.3% average improvement over ReAct across model scales
- **HotpotQA**: 7.3% average improvement over ReAct
- **Correction Efficiency**: 92% of successful corrections occur within first 2 attempts
- **Convergence**: Bounded corrections prevent infinite loops while maintaining high success rates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

For questions and support, please open an issue on GitHub or contact the authors.