"""
Basic usage examples for the ATCOT framework.
"""

import asyncio
import sys
import os

# Add the parent directory to the path to import atcot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import setup_atcot_framework


async def basic_math_example():
    """Example: Basic mathematical reasoning with corrections."""
    print("="*60)
    print("BASIC MATH EXAMPLE")
    print("="*60)
    
    framework = setup_atcot_framework()
    
    query = "Calculate 15% of 250, then add 30 to the result"
    
    result = await framework.execute(query)
    
    print(f"Query: {query}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Converged: {result.converged}")
    print(f"Total Corrections: {result.total_corrections}")
    print(f"Reasoning Steps: {result.statistics['reasoning_steps']}")
    print()


async def multi_step_example():
    """Example: Multi-step reasoning with tool coordination."""
    print("="*60)
    print("MULTI-STEP REASONING EXAMPLE")
    print("="*60)
    
    framework = setup_atcot_framework()
    
    query = "What is the capital of France, and what is the population of that city?"
    
    result = await framework.execute(query)
    
    print(f"Query: {query}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Tools Used: {list(result.statistics['tool_usage_distribution'].keys())}")
    print(f"Successful Tool Calls: {result.statistics['successful_tool_calls']}")
    print()


async def correction_example():
    """Example: Demonstrating correction mechanisms."""
    print("="*60)
    print("CORRECTION MECHANISM EXAMPLE")
    print("="*60)
    
    framework = setup_atcot_framework()
    
    # This query is designed to potentially trigger corrections
    query = "If I have $100 and spend 25% of it, then earn $20 more, how much do I have? Also calculate what 30% of my final amount would be."
    
    result = await framework.execute(query)
    
    print(f"Query: {query}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Total Corrections: {result.total_corrections}")
    print(f"Correction Types: {result.statistics['correction_types']}")
    print(f"Final Global Consistency: {result.statistics['final_global_consistency']}")
    print()


async def debug_example():
    """Example: Running with debug information."""
    print("="*60)
    print("DEBUG MODE EXAMPLE")
    print("="*60)
    
    framework = setup_atcot_framework()
    
    query = "Calculate the area of a circle with radius 5"
    
    def debug_callback(event_type: str, data: dict):
        print(f"[DEBUG] {event_type}: {data}")
    
    result = await framework.debug_execution(query, debug_callback)
    
    print(f"\nQuery: {query}")
    print(f"Final Answer: {result.final_answer}")
    print()


async def planning_example():
    """Example: Demonstrating adaptive planning."""
    print("="*60)
    print("ADAPTIVE PLANNING EXAMPLE")
    print("="*60)
    
    framework = setup_atcot_framework()
    
    query = "Research the current weather in New York, then suggest appropriate clothing for outdoor activities"
    
    result = await framework.execute(query)
    
    print(f"Query: {query}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Plan Steps: {result.statistics['total_plan_steps']}")
    print(f"Completed Steps: {result.statistics['completed_steps']}")
    print(f"Failed Steps: {result.statistics['failed_steps']}")
    print()


async def run_all_examples():
    """Run all examples in sequence."""
    print("ATCOT Framework Examples")
    print("========================")
    print()
    
    await basic_math_example()
    await multi_step_example() 
    await correction_example()
    await planning_example()
    await debug_example()
    
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(run_all_examples())
