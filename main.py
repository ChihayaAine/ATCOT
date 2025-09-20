"""
Main entry point for the ATCOT framework.

This module provides a simple interface for running ATCOT with different configurations
and demonstrates the complete framework usage.
"""

import asyncio
import logging
import argparse
from pathlib import Path

from atcot.core.framework import ATCOTFramework
from atcot.core.planning import AdaptivePlanner, LLMPlanGenerator
from atcot.core.execution import ToolAugmentedExecutor, LLMToolSelector
from atcot.core.correction import AdaptiveCorrectionMechanism, BackwardTraversalRevision
from atcot.tools.base import ToolRegistry
from atcot.tools.calculator import CalculatorTool
from atcot.tools.web_search import WebSearchTool
from atcot.tools.python_interpreter import PythonInterpreterTool
from atcot.tools.wikipedia import WikipediaTool
from atcot.utils.consistency import ConsistencyChecker
from atcot.utils.llm_interface import LLMInterfaceFactory
from atcot.utils.config import load_config

logger = logging.getLogger(__name__)


def setup_atcot_framework(config_path: str = None) -> ATCOTFramework:
    """
    Setup and configure the complete ATCOT framework.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured ATCOTFramework instance
    """
    # Load configuration
    config = load_config(config_path, use_env=True)
    
    logger.info(f"Setting up ATCOT framework with config: {config}")
    
    # Create LLM interface
    llm_interface = LLMInterfaceFactory.create_interface(
        provider=config.llm.provider,
        model_name=config.llm.model_name,
        api_key=config.llm.api_key
    )
    
    # Setup tool registry
    tool_registry = ToolRegistry()
    
    # Register available tools based on configuration
    if config.is_tool_enabled("calculator"):
        tool_registry.register_tool(CalculatorTool())
    
    if config.is_tool_enabled("web_search"):
        web_search_config = config.get_tool_config("web_search")
        search_engine = web_search_config.config_params.get("search_engine", "duckduckgo")
        tool_registry.register_tool(WebSearchTool(search_engine=search_engine))
    
    if config.is_tool_enabled("python_interpreter"):
        tool_registry.register_tool(PythonInterpreterTool())
    
    if config.is_tool_enabled("wikipedia"):
        wiki_config = config.get_tool_config("wikipedia")
        language = wiki_config.config_params.get("language", "en")
        tool_registry.register_tool(WikipediaTool(language=language))
    
    logger.info(f"Registered {len(tool_registry)} tools: {tool_registry.get_tool_names()}")
    
    # Create consistency checker
    consistency_checker = ConsistencyChecker(
        contradiction_threshold=config.correction.contradiction_threshold
    )
    
    # Create planning components
    plan_generator = LLMPlanGenerator(llm_interface)
    adaptive_planner = AdaptivePlanner(
        plan_generator=plan_generator,
        consistency_checker=consistency_checker,
        max_planning_attempts=config.planning.max_planning_attempts
    )
    
    # Create execution components
    tool_selector = LLMToolSelector(llm_interface)
    executor = ToolAugmentedExecutor(
        tool_registry=tool_registry,
        tool_selector=tool_selector,
        consistency_checker=consistency_checker,
        max_candidates_per_step=config.execution.max_candidates_per_step
    )
    
    # Create correction components
    revision_strategy = BackwardTraversalRevision(llm_interface, consistency_checker)
    correction_mechanism = AdaptiveCorrectionMechanism(
        consistency_checker=consistency_checker,
        revision_strategy=revision_strategy,
        contradiction_threshold=config.correction.contradiction_threshold,
        max_corrections=config.correction.max_corrections
    )
    
    # Create main framework
    framework = ATCOTFramework(
        llm_interface=llm_interface,
        tool_registry=tool_registry,
        consistency_checker=consistency_checker,
        adaptive_planner=adaptive_planner,
        executor=executor,
        correction_mechanism=correction_mechanism,
        max_correction_budget=config.correction.max_corrections
    )
    
    logger.info("ATCOT framework setup completed")
    return framework


async def run_atcot_query(framework: ATCOTFramework, query: str, debug: bool = False) -> None:
    """
    Run a single query through the ATCOT framework.
    
    Args:
        framework: Configured ATCOT framework
        query: Input query to process
        debug: Whether to enable debug mode
    """
    logger.info(f"Processing query: {query}")
    
    try:
        if debug:
            def debug_callback(event_type: str, data: dict):
                print(f"[DEBUG] {event_type}: {data}")
            
            result = await framework.debug_execution(query, debug_callback)
        else:
            result = await framework.execute(query)
        
        # Display results
        print("\n" + "="*80)
        print("ATCOT EXECUTION RESULTS")
        print("="*80)
        print(f"Query: {query}")
        print(f"Final Answer: {result.final_answer}")
        print(f"Converged: {result.converged}")
        print(f"Total Corrections: {result.total_corrections}")
        print(f"Execution Time: {result.execution_time_ms:.2f}ms")
        
        # Display statistics
        print("\nExecution Statistics:")
        for key, value in result.statistics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        print(f"Error: {e}")


async def interactive_mode(framework: ATCOTFramework) -> None:
    """
    Run ATCOT in interactive mode for multiple queries.
    
    Args:
        framework: Configured ATCOT framework
    """
    print("\nATCOT Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            elif query.lower() == 'help':
                print("Commands:")
                print("  help - Show this help")
                print("  info - Show framework information")
                print("  quit/exit - Exit interactive mode")
                print("  Or enter any query to process with ATCOT")
                continue
            elif query.lower() == 'info':
                info = framework.get_framework_info()
                print("Framework Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                continue
            elif not query:
                continue
            
            await run_atcot_query(framework, query, debug=False)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ATCOT: Adaptive Tool-Augmented Chain of Thought")
    parser.add_argument("--query", "-q", help="Query to process")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--examples", action="store_true", help="Show example queries")
    
    args = parser.parse_args()
    
    if args.examples:
        print("Example ATCOT Queries:")
        print("-" * 40)
        print("1. Mathematical: 'Calculate the compound interest on $1000 invested at 5% annually for 3 years'")
        print("2. Research: 'What is the current population of Tokyo and how has it changed over the last decade?'")
        print("3. Multi-step: 'Find the distance between New York and London, then calculate travel time at 500 mph'")
        print("4. Analysis: 'Compare the GDP of USA and China in 2023, then analyze the implications'")
        return
    
    # Setup framework
    try:
        framework = setup_atcot_framework(args.config)
    except Exception as e:
        logger.error(f"Failed to setup framework: {e}")
        print(f"Error setting up ATCOT framework: {e}")
        return
    
    # Run based on mode
    if args.interactive:
        asyncio.run(interactive_mode(framework))
    elif args.query:
        asyncio.run(run_atcot_query(framework, args.query, args.debug))
    else:
        # Default: run a sample query
        sample_query = "What is 25% of 200, and how much would that amount grow if invested at 8% annual interest for 2 years?"
        print(f"Running sample query: {sample_query}")
        asyncio.run(run_atcot_query(framework, sample_query, args.debug))


if __name__ == "__main__":
    main()
