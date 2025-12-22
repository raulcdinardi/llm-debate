#!/usr/bin/env python3
"""Debug script to inspect EVERYTHING returned by Tinker API calls.

Shows all fields from API responses to diagnose KV cache / extension property issues.
"""

import asyncio
import os
import sys
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
load_dotenv()

import tinker


async def main():
    print("=" * 80)
    print("TINKER API DEBUG - FULL RESPONSE INSPECTION")
    print("=" * 80)

    # Create client (same pattern as tinker_client.py)
    print("\n[1] Creating LoRA training client...")
    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async("debug_init")

    print(f"Service type: {type(service)}")
    print(f"Training client type: {type(training_client)}")
    print(f"Sampling client type: {type(sampling_client)}")

    client = sampling_client  # For sampling calls
    print(f"Client dir: {[attr for attr in dir(client) if not attr.startswith('_')]}")

    # Check if client has any useful attributes
    print("\nClient attributes:")
    for attr in dir(client):
        if not attr.startswith('_'):
            try:
                val = getattr(client, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

    # Simple generation
    print("\n" + "=" * 80)
    print("[2] Single sample_async call...")
    print("=" * 80)

    prompt = "What is 2 + 2?"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampling_params = tinker.SamplingParams(temperature=0.7, max_tokens=50)

    response = await client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )

    print(f"\nResponse type: {type(response)}")
    print(f"Response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")

    print("\n--- ALL RESPONSE ATTRIBUTES ---")
    for attr in dir(response):
        if not attr.startswith('_'):
            try:
                val = getattr(response, attr)
                if not callable(val):
                    print(f"{attr}:")
                    if isinstance(val, (list, tuple)) and len(val) > 20:
                        print(f"  (length={len(val)}) first 10: {val[:10]}")
                        print(f"  last 5: {val[-5:]}")
                    else:
                        pprint(val, indent=2)
                    print()
            except Exception as e:
                print(f"{attr}: <error accessing: {e}>\n")

    # Also inspect the sequence object
    seq = response.sequences[0]
    print(f"\nSequence type: {type(seq)}")
    print(f"Sequence dir: {[attr for attr in dir(seq) if not attr.startswith('_')]}")

    print("\n--- ALL SEQUENCE ATTRIBUTES ---")
    for attr in dir(seq):
        if not attr.startswith('_'):
            try:
                val = getattr(seq, attr)
                if not callable(val):
                    print(f"{attr}:")
                    if isinstance(val, (list, tuple)) and len(val) > 20:
                        print(f"  (length={len(val)}) first 10: {val[:10]}")
                        print(f"  last 5: {val[-5:]}")
                    else:
                        pprint(val, indent=2)
                    print()
            except Exception as e:
                print(f"{attr}: <error accessing: {e}>\n")

    # Extension test - multi-turn
    print("\n" + "=" * 80)
    print("[3] Extension property test (multi-turn)...")
    print("=" * 80)

    # R1
    r1_prompt = "You are a helpful assistant.\n\nUser: What is the capital of France?\nAssistant:"
    r1_prompt_tokens = tokenizer.encode(r1_prompt, add_special_tokens=False)
    print(f"\nR1 prompt: {len(r1_prompt)} chars, {len(r1_prompt_tokens)} tokens")

    r1_response = await client.sample_async(
        prompt=tinker.ModelInput.from_ints(r1_prompt_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(temperature=0.7, max_tokens=100),
    )

    # Extract response info
    r1_seq = r1_response.sequences[0]
    r1_completion_tokens = list(r1_seq.tokens)
    r1_completion_text = tokenizer.decode(r1_completion_tokens, skip_special_tokens=True)

    print(f"\nR1 Response - ALL FIELDS:")
    print(f"  Response type: {type(r1_response)}")
    print(f"  Response dir: {[a for a in dir(r1_response) if not a.startswith('_')]}")
    for attr in dir(r1_response):
        if not attr.startswith('_'):
            try:
                val = getattr(r1_response, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

    print(f"\nR1 Sequence[0] - ALL FIELDS:")
    print(f"  Sequence type: {type(r1_seq)}")
    print(f"  Sequence dir: {[a for a in dir(r1_seq) if not a.startswith('_')]}")
    for attr in dir(r1_seq):
        if not attr.startswith('_'):
            try:
                val = getattr(r1_seq, attr)
                if not callable(val):
                    if isinstance(val, (list, tuple)) and len(val) > 10:
                        print(f"  {attr}: len={len(val)}, first 5: {val[:5]}")
                    else:
                        print(f"  {attr}: {val}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

    # R2 - extend from R1 (critical for KV cache test)
    r2_prompt = r1_prompt + r1_completion_text + "\n\nUser: What about Germany?\nAssistant:"
    r2_prompt_tokens = tokenizer.encode(r2_prompt, add_special_tokens=False)

    print(f"\nR2 prompt: {len(r2_prompt)} chars, {len(r2_prompt_tokens)} tokens")
    print(f"R1 total was: {len(r1_prompt_tokens)} + {len(r1_completion_tokens)} = {len(r1_prompt_tokens) + len(r1_completion_tokens)} tokens")
    print(f"R2 prompt tokens ({len(r2_prompt_tokens)}) should > R1 total ({len(r1_prompt_tokens) + len(r1_completion_tokens)}) for extension")

    r2_response = await client.sample_async(
        prompt=tinker.ModelInput.from_ints(r2_prompt_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(temperature=0.7, max_tokens=100),
    )

    r2_seq = r2_response.sequences[0]
    r2_completion_tokens = list(r2_seq.tokens)

    print(f"\nR2 Response - ALL FIELDS:")
    for attr in dir(r2_response):
        if not attr.startswith('_'):
            try:
                val = getattr(r2_response, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

    print(f"\nR2 Sequence[0] - ALL FIELDS:")
    for attr in dir(r2_seq):
        if not attr.startswith('_'):
            try:
                val = getattr(r2_seq, attr)
                if not callable(val):
                    if isinstance(val, (list, tuple)) and len(val) > 10:
                        print(f"  {attr}: len={len(val)}, first 5: {val[:5]}")
                    else:
                        print(f"  {attr}: {val}")
            except Exception as e:
                print(f"  {attr}: <error: {e}>")

    # Token analysis
    print("\n" + "=" * 80)
    print("[4] Token count analysis for KV cache verification")
    print("=" * 80)

    r1_total = len(r1_prompt_tokens) + len(r1_completion_tokens)

    print(f"\nR1 prompt tokens: {len(r1_prompt_tokens)}")
    print(f"R1 completion tokens: {len(r1_completion_tokens)}")
    print(f"R1 total: {r1_total}")
    print(f"\nR2 prompt tokens: {len(r2_prompt_tokens)}")
    print(f"R2 completion tokens: {len(r2_completion_tokens)}")
    print(f"\nExpected if extending: R2_prompt should be R1_total + continuation_tokens")
    print(f"Continuation would be: {len(r2_prompt_tokens) - r1_total} tokens")

    if len(r2_prompt_tokens) > r1_total:
        print("\n✓ Extension property appears to hold (R2 prompt > R1 total)")
        print("  (But this doesn't guarantee KV cache is being billed correctly)")
    else:
        print("\n✗ Extension property may NOT hold (R2 prompt <= R1 total)")

    # Check for any billing/usage fields
    print("\n" + "=" * 80)
    print("[5] Looking for billing/usage fields...")
    print("=" * 80)

    billing_keywords = ['cost', 'bill', 'usage', 'price', 'credit', 'charge', 'cache', 'kv', 'prefix']

    print("\nSearching in response object:")
    for attr in dir(r1_response):
        attr_lower = attr.lower()
        if any(kw in attr_lower for kw in billing_keywords):
            try:
                val = getattr(r1_response, attr)
                print(f"  Found: {attr} = {val}")
            except:
                pass

    print("\nSearching in sequence object:")
    for attr in dir(r1_seq):
        attr_lower = attr.lower()
        if any(kw in attr_lower for kw in billing_keywords):
            try:
                val = getattr(r1_seq, attr)
                print(f"  Found: {attr} = {val}")
            except:
                pass

    print("\nSearching in sampling_client:")
    for attr in dir(sampling_client):
        attr_lower = attr.lower()
        if any(kw in attr_lower for kw in billing_keywords):
            try:
                val = getattr(sampling_client, attr)
                if not callable(val):
                    print(f"  Found: {attr} = {val}")
            except:
                pass

    print("\nSearching in training_client:")
    for attr in dir(training_client):
        attr_lower = attr.lower()
        if any(kw in attr_lower for kw in billing_keywords):
            try:
                val = getattr(training_client, attr)
                if not callable(val):
                    print(f"  Found: {attr} = {val}")
            except:
                pass

    # Raw dict check
    print("\n" + "=" * 80)
    print("[6] Checking if response has __dict__ or can be converted to dict...")
    print("=" * 80)

    print("\nResponse object:")
    if hasattr(r1_response, '__dict__'):
        print("  __dict__:")
        pprint(r1_response.__dict__)

    if hasattr(r1_response, '_asdict'):
        print("  _asdict():")
        pprint(r1_response._asdict())

    if hasattr(r1_response, 'to_dict'):
        print("  to_dict():")
        pprint(r1_response.to_dict())

    print("\nSequence object:")
    if hasattr(r1_seq, '__dict__'):
        print("  __dict__:")
        for k, v in r1_seq.__dict__.items():
            if isinstance(v, (list, tuple)) and len(v) > 10:
                print(f"    {k}: len={len(v)}")
            else:
                print(f"    {k}: {v}")

    if hasattr(r1_seq, '_asdict'):
        print("  _asdict():")
        pprint(r1_seq._asdict())

    if hasattr(r1_seq, 'to_dict'):
        print("  to_dict():")
        pprint(r1_seq.to_dict())

    # Check tinker module
    print("\n" + "=" * 80)
    print("[7] Tinker module inspection...")
    print("=" * 80)
    print(f"tinker module dir: {[x for x in dir(tinker) if not x.startswith('_')]}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
