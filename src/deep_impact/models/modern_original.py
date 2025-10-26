import os
import string
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Set

import numpy as np
import tokenizers
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.modernbert.modeling_modernbert import ModernBertModel, ModernBertPreTrainedModel

from src.utils.checkpoint import ModelCheckpoint


class ModernDeepImpact(ModernBertPreTrainedModel):
    """
    ModernBERT-based DeepImpact model for sparse learned retrieval.

    Key differences from BERT-based DeepImpact:
    - Uses ModernBertModel (22 layers, RoPE, GeGLU, Flash Attention)
    - Increased context length: 1024 tokens (vs 512 for BERT)
    - Does NOT use token_type_ids (ModernBERT doesn't support them)
    - Uses Alibaba's tokenizer (vocab: 50,280 vs BERT's 30,522)
    - Initialized from Alibaba's gte-modernbert-base (fine-tuned for retrieval)
    """

    max_length = 1024  # Increased from 512 - ModernBERT supports up to 8192
    # CRITICAL: Use the same tokenizer as the model we load (Alibaba's gte-modernbert-base)
    # Vocab sizes MUST match: Alibaba has 50,280 tokens vs answerdotai's 50,368
    tokenizer = tokenizers.Tokenizer.from_pretrained('Alibaba-NLP/gte-modernbert-base')
    tokenizer.enable_truncation(max_length)
    punctuation = set(string.punctuation)

    def __init__(self, config):
        super(ModernDeepImpact, self).__init__(config)

        # ModernBERT encoder instead of BERT
        self.modernbert = ModernBertModel(config)

        # Same impact score encoder (768 → 1 with ReLU)
        # This stays the same because ModernBERT-base has hidden_size=768 like BERT-base
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )

        # ModernBERT uses post_init() instead of init_weights()
        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,  # Made optional - ModernBERT ignores this
    ) -> torch.Tensor:
        """
        Forward pass through ModernBERT and impact score encoder.

        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids (IGNORED - kept for backward compatibility)
        :return: Batch of impact scores
        """
        # Note: token_type_ids is ignored - ModernBERT doesn't use segment embeddings
        bert_output = self._get_bert_output(input_ids, attention_mask, token_type_ids)
        return self._get_term_impact_scores(bert_output.last_hidden_state)

    def _get_bert_output(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,  # Optional now
            output_attentions: Optional[bool] = None,
    ):
        """
        Get ModernBERT encoder output.

        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids (IGNORED by ModernBERT)
        :param output_attentions: Whether to output attentions
        :return: ModernBERT outputs
        """
        # ModernBERT doesn't accept token_type_ids, so we don't pass it
        return self.modernbert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

    def _get_term_impact_scores(
            self,
            last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project hidden states to impact scores.

        :param last_hidden_state: Last hidden state from ModernBERT
        :return: Impact scores
        """
        return self.impact_score_encoder(last_hidden_state)

    @classmethod
    def process_query_and_document(cls, query: str, document: str, max_length: Optional[int] = None) -> \
            Tuple[tokenizers.Encoding, torch.Tensor]:
        """
        Process query and document to feed to the model.

        :param query: Query string
        :param document: Document string
        :param max_length: Max number of tokens to process
        :return: Tuple: Document Tokens, Mask with 1s corresponding to first tokens of document terms in the query
        """
        query_terms = cls.process_query(query)
        encoded, term_to_token_index = cls.process_document(document)

        return encoded, cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length)

    @classmethod
    def get_query_document_token_mask(cls, query_terms: Set[str], term_to_token_index: Dict[str, int],
                                      max_length: Optional[int] = None) -> torch.Tensor:
        """
        Create binary mask for query-matching document terms.

        :param query_terms: Set of query terms
        :param term_to_token_index: Mapping from terms to token indices
        :param max_length: Max sequence length
        :return: Binary mask tensor
        """
        if max_length is None:
            max_length = cls.max_length

        mask = np.zeros(max_length, dtype=bool)
        token_indices_of_matching_terms = [v for k, v in term_to_token_index.items() if k in query_terms]
        mask[token_indices_of_matching_terms] = True

        return torch.from_numpy(mask)

    @classmethod
    def process_query(cls, query: str) -> Set[str]:
        """
        Tokenize and extract query terms.

        Note: ModernBERT uses byte-level BPE with 'Ġ' (U+0120) marking word boundaries.
        We strip this prefix to ensure query terms match document terms correctly.

        :param query: Query string
        :return: Set of query terms (excluding punctuation, with Ġ prefix stripped)
        """
        query = cls.tokenizer.normalizer.normalize_str(query)
        # Strip Ġ prefix from byte-level BPE tokens to normalize terms
        return set(filter(lambda x: x not in cls.punctuation,
                          map(lambda x: x[0].lstrip('Ġ'), cls.tokenizer.pre_tokenizer.pre_tokenize_str(query))))

    @classmethod
    def process_document(cls, document: str) -> Tuple[tokenizers.Encoding, Dict[str, int]]:
        """
        Encodes the document and maps each unique term (non-punctuation) to its corresponding first token's index.

        Note: ModernBERT uses byte-level BPE with 'Ġ' (U+0120) marking word boundaries.
        We encode the document directly (not with is_pretokenized=True) to avoid creating
        extra boundary tokens that would break the term-to-token mapping.

        :param document: Document string
        :return: Tuple: Encoded document, Dict mapping unique non-punctuation document terms to first token index
        """
        # Normalize the document text
        document = cls.tokenizer.normalizer.normalize_str(document)

        # Encode directly - do NOT use is_pretokenized=True!
        # Using is_pretokenized with Ġ-prefixed terms creates extra boundary tokens (Ä, ł, etc.)
        # that break the term-to-token mapping.
        encoded = cls.tokenizer.encode(document)

        # Build term-to-token mapping by identifying word boundaries
        # In byte-level BPE, words are marked by:
        # 1. Tokens starting with 'Ġ' (word boundary)
        # 2. The first token after [CLS]
        # 3. Subwords continue the current word (no Ġ prefix)

        filtered_term_to_token_index = {}
        current_term_tokens = []
        current_term_start_idx = None

        # Process all tokens except [CLS]
        # We'll skip special tokens ([SEP], [PAD], etc.) in the loop
        tokens = encoded.tokens

        for i, token in enumerate(tokens):
            # Skip special tokens (anything in square brackets)
            if token.startswith('[') and token.endswith(']'):
                # Before skipping, save any accumulated term
                if current_term_tokens:
                    term = ''.join(current_term_tokens).lstrip('Ġ')
                    if term and term not in cls.punctuation and current_term_start_idx is not None:
                        if term not in filtered_term_to_token_index:
                            filtered_term_to_token_index[term] = current_term_start_idx
                    # Reset accumulator
                    current_term_tokens = []
                    current_term_start_idx = None
                continue

            # Check if this is the start of a new word
            is_new_word = (
                token.startswith('Ġ') or  # Explicit word boundary
                (i > 0 and tokens[i-1].startswith('['))  # First token after special token
            )

            if is_new_word:
                # Save previous term if it exists
                if current_term_tokens:
                    # Reconstruct term from accumulated tokens
                    term = ''.join(current_term_tokens).lstrip('Ġ')

                    # Only add non-empty, non-punctuation terms
                    if term and term not in cls.punctuation and current_term_start_idx is not None:
                        # Check if term isn't already in dict (keep first occurrence)
                        if term not in filtered_term_to_token_index:
                            filtered_term_to_token_index[term] = current_term_start_idx

                # Start accumulating new term
                current_term_tokens = [token]
                current_term_start_idx = i
            else:
                # This is a subword continuation of the current term
                current_term_tokens.append(token)

        # Don't forget to add the last term (if not already saved by special token logic)
        if current_term_tokens:
            term = ''.join(current_term_tokens).lstrip('Ġ')
            if term and term not in cls.punctuation and current_term_start_idx is not None:
                if term not in filtered_term_to_token_index:
                    filtered_term_to_token_index[term] = current_term_start_idx

        return encoded, filtered_term_to_token_index

    @classmethod
    def load(cls, checkpoint_path: Optional[Union[str, Path]] = None):
        """
        Load ModernDeepImpact model.

        By default, initializes from Alibaba's gte-modernbert-base which is:
        - Pre-trained ModernBERT
        - Fine-tuned for text embedding and retrieval tasks
        - Better starting point than vanilla ModernBERT for our sparse retrieval task

        :param checkpoint_path: Optional path to a trained checkpoint
        :return: Loaded model
        """
        # Use Alibaba's retrieval-optimized ModernBERT as base
        model = cls.from_pretrained('Alibaba-NLP/gte-modernbert-base')

        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                # Load our trained weights on top of the base model
                ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
            else:
                # checkpoint_path is a HuggingFace model ID
                model = cls.from_pretrained(checkpoint_path)

        # Configure tokenizer for our max_length
        cls.tokenizer.enable_truncation(max_length=cls.max_length, strategy='longest_first')
        cls.tokenizer.enable_padding(length=cls.max_length)

        return model

    @staticmethod
    def compute_term_impacts(
            documents_term_to_token_index_map: List[Dict[str, int]],
            outputs: torch.Tensor,
    ) -> List[List[Tuple[str, float]]]:
        """
        Computes the impact scores of each term in each document.

        :param documents_term_to_token_index_map: List of dictionaries mapping each unique term to its first token index
        :param outputs: Batch of model outputs
        :return: Batch of lists of tuples of document terms and their impact scores
        """
        impact_scores = outputs.squeeze(-1).cpu().numpy()

        term_impacts = []
        for i, term_to_token_index_map in enumerate(documents_term_to_token_index_map):
            term_impacts.append([
                (term, impact_scores[i][token_index])
                for term, token_index in term_to_token_index_map.items()
            ])

        return term_impacts

    def get_impact_scores(self, document: str) -> List[Tuple[str, float]]:
        """
        Get impact scores for each term in the document.

        :param document: Document string
        :return: List of tuples of document terms and their impact scores
        """
        encoded, term_to_token_index = self.process_document(document)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long).to(self.device)

        # Note: We don't create token_type_ids for ModernBERT
        # If token_type_ids exist in encoded object, we can pass None anyway

        with torch.no_grad():
            outputs = self(input_ids, attention_mask, token_type_ids=None)

        return self.compute_term_impacts([term_to_token_index], outputs)[0]

    def get_impact_scores_batch(self, documents: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Get impact scores for each term in each document in batches.

        :param documents: List of document strings
        :return: List of lists of tuples of document terms and their impact scores
        """
        # Process all documents in batch
        encoded_docs = []
        term_to_token_maps = []
        for doc in documents:
            encoded, term_map = self.process_document(doc)
            encoded_docs.append(encoded)
            term_to_token_maps.append(term_map)

        # Create batched tensors
        input_ids = torch.tensor([enc.ids for enc in encoded_docs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([enc.attention_mask for enc in encoded_docs], dtype=torch.long).to(self.device)

        # ModernBERT doesn't use token_type_ids, so we pass None

        # Get model outputs for full batch
        with torch.no_grad():
            outputs = self(input_ids, attention_mask, token_type_ids=None)

        # Compute impact scores for all documents
        return self.compute_term_impacts(term_to_token_maps, outputs)
