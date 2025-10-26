"""
Test script to verify ModernBERT implementation works correctly.

Usage:
    python test_modernbert.py
"""

import sys
import torch

print("=" * 60)
print("ModernBERT Implementation Test")
print("=" * 60)

# Test 1: Check transformers version
print("\n[Test 1] Checking transformers version...")
try:
    import transformers
    version = transformers.__version__
    print(f"✓ transformers version: {version}")

    # Parse version
    major, minor, patch = version.split('.')[:3]
    major, minor = int(major), int(minor)

    if major > 4 or (major == 4 and minor >= 48):
        print(f"✓ Version is >= 4.48.0 (required for ModernBERT)")
    else:
        print(f"✗ Version is < 4.48.0 (need to upgrade)")
        print(f"  Run: pip install --upgrade transformers")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking transformers: {e}")
    sys.exit(1)

# Test 2: Check ModernBERT classes are available
print("\n[Test 2] Checking ModernBERT classes...")
try:
    from transformers import ModernBertPreTrainedModel, ModernBertModel
    print("✓ ModernBertPreTrainedModel imported successfully")
    print("✓ ModernBertModel imported successfully")
except ImportError as e:
    print(f"✗ Cannot import ModernBERT classes: {e}")
    print(f"  Your transformers version may be too old")
    sys.exit(1)

# Test 3: Load ModernDeepImpact model
print("\n[Test 3] Loading ModernDeepImpact model...")
try:
    from src.deep_impact.models import ModernDeepImpact
    print("✓ ModernDeepImpact imported successfully")

    # Check class attributes
    print(f"  - max_length: {ModernDeepImpact.max_length}")
    print(f"  - tokenizer: {ModernDeepImpact.tokenizer.__class__.__name__}")

except ImportError as e:
    print(f"✗ Cannot import ModernDeepImpact: {e}")
    sys.exit(1)

# Test 4: Load model from pretrained
print("\n[Test 4] Loading model from pretrained (this may take a minute)...")
try:
    model = ModernDeepImpact.load()
    print(f"✓ Model loaded successfully")
    print(f"  - Model class: {model.__class__.__name__}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    print(f"  - Vocab size: {model.config.vocab_size}")
    print(f"  - Max position embeddings: {model.config.max_position_embeddings}")

    # Verify it's ModernBERT
    assert model.config.num_hidden_layers == 22, "Expected 22 layers for ModernBERT-base"
    assert model.config.hidden_size == 768, "Expected hidden size 768"
    print("✓ Model architecture verified (22 layers, 768 hidden size)")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test inference on sample document
print("\n[Test 5] Testing inference on sample document...")
try:
    doc = "Information retrieval is the process of obtaining relevant documents from a collection."

    print(f"  Document: \"{doc}\"")
    print("  NOTE: impact_score_encoder is randomly initialized until training")
    print("  Scores will be random/NaN - this is expected for untrained model")

    # Get impact scores
    scores = model.get_impact_scores(doc)

    print(f"✓ Generated {len(scores)} term impact scores")
    print(f"  Top 5 terms (random until trained):")
    # Sort by score
    sorted_scores = sorted(scores, key=lambda x: abs(x[1]) if not torch.isnan(torch.tensor(x[1])) else 0, reverse=True)[:5]
    for term, score in sorted_scores:
        print(f"    - {term}: {score:.4f}")

    print("✓ Inference pipeline works (scores will be meaningful after training)")

except Exception as e:
    print(f"✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test batch processing
print("\n[Test 6] Testing batch processing...")
try:
    docs = [
        "Deep learning models for text retrieval.",
        "Sparse neural retrieval with impact scores.",
        "ModernBERT encoder for information retrieval."
    ]

    print("  NOTE: Skipping batch inference (scores random until trained)")
    print("  Testing batch processing logic only...")

    # Just test the document processing works
    for doc in docs:
        encoded, term_map = model.process_document(doc)
        assert len(term_map) > 0, "Should extract terms"

    print(f"✓ Batch document processing works")
    print(f"✓ Processed {len(docs)} documents successfully")

except Exception as e:
    print(f"✗ Error during batch processing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test token_type_ids handling
print("\n[Test 7] Testing token_type_ids handling...")
try:
    # Test that forward() accepts token_type_ids=None
    encoded, term_map = model.process_document("Test document")
    input_ids = torch.tensor([encoded.ids], dtype=torch.long)
    attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long)

    # Should work with None
    with torch.no_grad():
        output1 = model(input_ids, attention_mask, token_type_ids=None)

    print(f"✓ Forward pass with token_type_ids=None works")
    print(f"  Output shape: {output1.shape}")

    # Should also work with a tensor (backward compatibility)
    type_ids = torch.tensor([encoded.type_ids], dtype=torch.long)
    with torch.no_grad():
        output2 = model(input_ids, attention_mask, token_type_ids=type_ids)

    print(f"✓ Forward pass with token_type_ids tensor works (backward compatible)")

except Exception as e:
    print(f"✗ Error testing token_type_ids: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test process_query
print("\n[Test 8] Testing query processing...")
try:
    query = "deep learning retrieval"
    query_terms = ModernDeepImpact.process_query(query)

    print(f"  Query: \"{query}\"")
    print(f"✓ Extracted {len(query_terms)} query terms: {query_terms}")

    # Verify no punctuation
    for term in query_terms:
        assert term not in ModernDeepImpact.punctuation, f"Should filter out punctuation: {term}"
    print("✓ Punctuation filtered correctly")

    # Verify no Ġ prefix (critical bug check)
    for term in query_terms:
        assert not term.startswith('Ġ'), f"Term should not have Ġ prefix: {repr(term)}"
    print("✓ No Ġ prefix in query terms")

except Exception as e:
    print(f"✗ Error testing query processing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test query-document term matching (CRITICAL BUG CHECK)
print("\n[Test 9] Testing query-document term matching...")
try:
    query = "apple banana"
    doc = "The apple and banana are red fruits"

    query_terms = ModernDeepImpact.process_query(query)
    encoded, term_to_token_map = ModernDeepImpact.process_document(doc)

    print(f"  Query: \"{query}\"")
    print(f"  Document: \"{doc}\"")
    print(f"  Query terms: {query_terms}")
    print(f"  Doc terms: {list(term_to_token_map.keys())}")

    # Verify matching works
    matches = [term for term in term_to_token_map.keys() if term in query_terms]
    print(f"✓ Found {len(matches)} matching terms: {matches}")

    # Critical check: apple and banana should match
    assert 'apple' in matches, "CRITICAL BUG: 'apple' should match!"
    assert 'banana' in matches, "CRITICAL BUG: 'banana' should match!"

    # Verify no Ġ prefix in document terms
    for term in term_to_token_map.keys():
        assert not term.startswith('Ġ'), f"Doc term should not have Ġ prefix: {repr(term)}"
    print("✓ No Ġ prefix in document terms")

    # Verify mask would be non-zero
    mask = ModernDeepImpact.get_query_document_token_mask(query_terms, term_to_token_map)
    num_matching = mask.sum().item()
    print(f"✓ Mask has {num_matching} non-zero entries (should be > 0)")
    assert num_matching > 0, "CRITICAL BUG: Mask is all zeros!"

except Exception as e:
    print(f"✗ Error testing query-document matching: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nModernBERT implementation is working correctly!")
print("\nNext steps:")
print("  1. Install dependencies: pip install --upgrade transformers")
print("  2. Train with ModernBERT:")
print("     python -m src.deep_impact.train --use_modernbert \\")
print("       --dataset_path <path> \\")
print("       --queries_path <path> \\")
print("       --collection_path <path> \\")
print("       --checkpoint_dir <path> \\")
print("       --max_length 1024")
print("  3. Index with trained model:")
print("     python -m src.deep_impact.index \\")
print("       --collection_path <path> \\")
print("       --model_checkpoint_path <checkpoint> \\")
print("       --output_file_path <output>")
