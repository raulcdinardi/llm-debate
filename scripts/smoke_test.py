"""
Smoke test for Tinker API connection.

Verifies API key works and we can sample from a model.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 60)
    print("SMOKE TEST: Tinker API Connection")
    print("=" * 60)

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("ERROR: TINKER_API_KEY not set in .env")
        return False

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    print("\n[1/4] Importing tinker...")
    import tinker
    from tinker_cookbook.renderers import get_renderer, Qwen3InstructRenderer

    print("[2/4] Creating training client (to get tokenizer)...")
    service = tinker.ServiceClient()

    # Use a small model from Tinker's lineup
    model_name = "Qwen/Qwen3-4B-Instruct-2507"  # Compact size
    print(f"  Model: {model_name}")

    training_client = service.create_lora_training_client(base_model=model_name)
    tokenizer = training_client.get_tokenizer()

    print("[3/4] Setting up renderer and sampling client...")
    renderer = Qwen3InstructRenderer(tokenizer)
    sampling_client = training_client.save_weights_and_get_sampling_client("smoke_test")

    print("[4/4] Testing sampling...")
    # Simple test prompt
    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]
    model_input = renderer.build_generation_prompt(messages)
    stop_strings = renderer.get_stop_sequences()

    print("  Sending request...")
    result = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.7,
            stop=stop_strings if isinstance(stop_strings[0], str) else None,
        ),
    )

    # Get the result
    sample_response = result.result()
    output_tokens = sample_response.sequences[0].tokens
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(f"  Response: {response}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED - Tinker API working!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
