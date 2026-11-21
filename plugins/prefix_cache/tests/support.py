"""Shared, finite-vocabulary tokenizers for offline plugin tests."""


class BoundaryMergeTokenizer:
    """Reversible ASCII tokenizer with one greedy token spanning a text boundary."""

    all_special_ids = list(range(32))

    def __init__(self, merge_text=" s"):
        if not merge_text or not merge_text.isascii():
            raise ValueError("merge_text must be non-empty ASCII")
        self.merge_text = merge_text

    def __len__(self):
        return 129

    def encode(self, text, add_special_tokens=False):
        if not text.isascii():
            raise ValueError("BoundaryMergeTokenizer accepts ASCII input only")
        result, index = [], 0
        while index < len(text):
            if text.startswith(self.merge_text, index):
                result.append(128)
                index += len(self.merge_text)
            else:
                result.append(ord(text[index]))
                index += 1
        return result

    def decode(self, token_ids, skip_special_tokens=False):
        if any(not isinstance(token_id, int) or not 0 <= token_id < len(self)
               for token_id in token_ids):
            raise ValueError("token ID is outside the ASCII fixture vocabulary")
        return "".join(self.merge_text if token_id == 128 else chr(token_id)
                       for token_id in token_ids)
