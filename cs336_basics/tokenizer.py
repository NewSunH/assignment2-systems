from .bpe import BpeTokenizer
from .bpe import _split_special_tokens
import json
from typing import Iterable, Iterator
import regex as re
from collections import OrderedDict


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Precompute frequently-used structures for speed.
        self._byte_to_id = {v: k for k, v in self.vocab.items()}
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self._pat = re.compile(BpeTokenizer.PAT)

        self._special_re = None
        if self.special_tokens:
            toks_sorted = sorted(self.special_tokens, key=len, reverse=True)
            toks = "(" + "|".join(re.escape(t) for t in toks_sorted) + ")"
            self._special_re = re.compile(toks)

        # Simple bounded cache for BPE results on frequently repeating pretokens.
        self._bpe_cache: OrderedDict[bytes, tuple[bytes, ...]] = OrderedDict()
        self._bpe_cache_max = 50_000

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # Our BPE serialization follows GPT-2 style:
        # - vocab.json maps an encoded string (representing arbitrary bytes) -> token_id
        # - merges.txt stores two encoded strings per line
        # We must decode the encoded strings back to raw bytes to build the runtime vocab.
        encoder = BpeTokenizer._bytes_to_unicode()
        decoder = {v: k for k, v in encoder.items()}

        def decode_token_str(token_str: str) -> bytes:
            return bytes(decoder[ch] for ch in token_str)

        with open(vocab_filepath, "r", encoding="utf-8") as vocab_f:
            raw_vocab: dict[str, int] = json.load(vocab_f)

        vocab: dict[int, bytes] = {}
        for token_str, token_id in raw_vocab.items():
            vocab[int(token_id)] = decode_token_str(token_str)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((decode_token_str(a), decode_token_str(b)))
        return cls(vocab, merges, special_tokens)

    @classmethod
    def from_bpe_cls(cls, bpe_tokenizer: BpeTokenizer):
        return cls(
            bpe_tokenizer.vocab, bpe_tokenizer.merges, bpe_tokenizer.special_tokens
        )

    def _bpe_merge(
        self,
        seq: list[bytes],
        merge_rank: dict[tuple[bytes, bytes], int],
    ) -> list[bytes]:
        while True:
            best_pair = None
            best_rank = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                r = merge_rank.get(pair)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            ab = a + b
            new_seq: list[bytes] = []
            i = 0
            n = len(seq)
            while i < n:
                if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(ab)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq

        return seq

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        if self._special_re is not None:
            parts = self._special_re.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if self.special_tokens and part in self.special_tokens:
                ids.append(self._byte_to_id[part.encode("utf-8")])
                continue

            # GPT-2 tokenization: pretokenize with the regex, then apply BPE within each pretoken.
            for pretoken in self._pat.findall(part):
                bs = pretoken.encode("utf-8", errors="ignore")

                cached = self._bpe_cache.get(bs)
                if cached is None:
                    seq = [bytes([b]) for b in bs]
                    merged = tuple(self._bpe_merge(seq, self._merge_rank))
                    # maintain a bounded LRU-ish cache
                    self._bpe_cache[bs] = merged
                    if len(self._bpe_cache) > self._bpe_cache_max:
                        self._bpe_cache.popitem(last=False)
                    cached = merged
                else:
                    # mark as recently used
                    self._bpe_cache.move_to_end(bs)

                ids.extend(self._byte_to_id[tok] for tok in cached)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        bytes_seq = [self.vocab[token_id] for token_id in ids]
        full_bytes = b"".join(bytes_seq)
        decoded_string = full_bytes.decode("utf-8", errors="ignore")
        return decoded_string
