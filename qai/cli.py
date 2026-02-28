"""Interactive CLI for the QRL quantum AI loop.

Usage:
    cd quantum-relational-language
    source .venv/bin/activate
    PYTHONPATH=src python -m qai.cli

    # or with a specific model:
    PYTHONPATH=src python -m qai.cli --model deepseek-coder-v2:16b
    PYTHONPATH=src python -m qai.cli --provider together  # requires TOGETHER_API_KEY
    PYTHONPATH=src python -m qai.cli --explain-claude     # requires ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import sys

from .loop import QuantumAILoop
from .providers import OllamaProvider, TogetherAIProvider, ClaudeProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="QRL Quantum AI — ask quantum questions")
    parser.add_argument("--model", default="deepseek-coder-v2:16b", help="Ollama model for code gen")
    parser.add_argument("--provider", choices=["ollama", "together"], default="ollama")
    parser.add_argument("--explain-claude", action="store_true",
                        help="Use Claude (haiku) for explanation step (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--verbose", action="store_true", help="Show generated QRL code")
    args = parser.parse_args()

    if args.provider == "together":
        code_provider = TogetherAIProvider()
    else:
        code_provider = OllamaProvider(model=args.model)

    explain_provider = ClaudeProvider() if args.explain_claude else code_provider
    loop = QuantumAILoop(code_provider=code_provider, explain_provider=explain_provider)

    print("QRL Quantum AI")
    print(f"Model: {args.model} via {args.provider}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Bye.")
            sys.exit(0)

        print("⏳ Thinking...", flush=True)
        try:
            answer = loop.ask(question, verbose=args.verbose)
            print(f"\n💡 {answer}\n")
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")


if __name__ == "__main__":
    main()
