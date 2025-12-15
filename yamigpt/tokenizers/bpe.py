"""Byte-Pair Encoding tokenizer"""

import unicodedata
import json
from collections import Counter
from pathlib import Path
from typing import Any
from .abc import AbstractTokenizer

type PairOfTokens = tuple[int, int]

class BPETokenizer(AbstractTokenizer[int]):
    """Byte-Pair Encoding tokenizer"""

    def __init__(
        self,
        vocab_size: int,
    ):
        if vocab_size < 256:
            raise ValueError("Vocab size should be at least 256")
        self._vocab_size = vocab_size
        # list of tuples with structure:
        # (pair_to_merge, id_to_merge_with)
        self._merges: list[
            tuple[PairOfTokens, int]
        ] = []
        # vocabulary: token -> id
        self._vocab: dict[tuple[int, ...], int] = {}
        # inverse of vocabulary: id -> token
        self._id_to_token: dict[int, tuple[int, ...]] = {}

        # first 255 unicode bytes
        for i in range(256):
            self._vocab[(i,)] = i
            self._id_to_token[i] = (i,)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocab(self) -> dict[tuple[int, ...], int]:
        return self._vocab

    @property
    def merges(self) -> list[tuple[PairOfTokens, int]]:
        return self._merges

    def tokenize(
        self,
        text: str,
        **kwargs: Any
    ) -> list[int]:
        """Tokenizes text using BPE algorithm

        :param text: text to tokenize.
        :param kwargs: additional params. Unused.

        :return: list of tokens as bytes.
        """

        tokens = self.preprocess(text)

        for (a, b), new_token in self._merges:
            i = 0
            result = []

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == (a,b):
                    result.append(new_token)
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1

            tokens = result

        return tokens

    def fit(self, corpus: list[str], **kwargs: Any) -> None:
        """Fits tokenizer for specific corpus of texts

        :param corpus: corpus of texts to train on.
        :param kwargs: additional params. Unused.

        :return: Nothing, updates tokenizer state.
        """
        seqs: list[list[int]] = []
        for text in corpus:
            seqs.append(self.preprocess(text))

        next_id = 256 # first 255 bytes are reserved for unicode

        i = 0
        while next_id < self._vocab_size:
            print("Iteration", i)
            print(f"\tCurrent vocab size: {next_id + 1}/{self._vocab_size}")
            stats = self._get_pair_statistics(seqs)


            if not stats:
                print("No more pairs left. Vocab size:", next_id)
                self._vocab_size = next_id
                break

            pair = max(stats, key=stats.get)

            seqs = self._merge_pair(seqs, pair, next_id)

            # updating mappings
            new_token = self._id_to_token[pair[0]] + self._id_to_token[pair[1]]

            self._vocab[new_token] = next_id
            self._id_to_token[next_id] = new_token
            self._merges.append((pair, next_id))

            next_id += 1
            i += 1

    def decode(
        self,
        tokens: list[int],
    ) -> str:
        """Decodes tokens back to text"""
        bytes_seq: list[int] = []

        for token in tokens:
            bytes_seq.extend(self._id_to_token[token])

        return bytes(bytes_seq).decode("utf-8", "replace")

    def save(self, checkpoint_path: str | Path) -> None:
        """Saves tokenizer state as json

        :param checkpoint_path: path to file to save.

        :return: only saves the state
        """
        state = {
            "vocab_size": self._vocab_size,
            "merges": [
                {
                    "pair": [a, b],
                    "new_id": new_id
                }
                for (a,b), new_id in self._merges
            ],
            "vocab": {
                ",".join(map(str, token)): idx
                for token, idx in self._vocab.items()
            },
            "id_to_token": {
                str(idx): list(token)
                for idx, token in self._id_to_token.items()
            }
        }

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        print("State saved to", checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> BPETokenizer:
        """Loads tokenizer from json checkpoint"""
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data.get("vocab_size"))

        tokenizer._merges = [
            ((m["pair"][0], m["pair"][1]), m["new_id"])
            for m in data.get("merges")
        ]

        tokenizer._vocab = {
            tuple(map(int, k.split(","))): v
            for k, v in data.get("vocab").items()
        }

        tokenizer._id_to_token = {
            int(idx): tuple(token)
            for idx, token in data.get("id_to_token").items()
        }

        return tokenizer

    def _get_pair_statistics(self, seqs: list[list[int]]) -> Counter | None:
        """Makes statistics of pairs in tokens list

        :param tokens: tokens to get statistics of.
        
        :return: counter of pairs.
        """
        pairs: list[PairOfTokens] = []

        for seq in seqs:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])

                pairs.append(pair)

        if not pairs:
            return None

        count = Counter(pairs)

        return count

    def _merge_pair(
        self,
        seqs: list[list[int]],
        pair: PairOfTokens,
        next_id: int
    ) -> list[list[int]]:
        """Merges pair of tokens in tokens list

        :param tokens: initial list of tokens
            where we want to merge pair.
        :param pair: pair to merge.
        :param next_id: id of token to replace pair with.

        :return: list of merged tokens
        """
        new_seqs: list[list[int]] = []

        for seq in seqs:
            result: list[int] = []
            i = 0

            while i < len(seq):

                if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                    result.append(next_id)

                    i += 2
                else:
                    result.append(seq[i])

                    i += 1
            new_seqs.append(result)

        return new_seqs

    def preprocess(self, text: str) -> list[int]:
        """Preprocess text for tokenization

        Performs unicode normalization and utf-8 encoding.

        :param text: text to preprocess

        :return: list of initial tokens
        """

        text = unicodedata.normalize("NFKC", text)
        tokens = list(text.encode("utf-8"))

        return tokens

    def __len__(self) -> int:
        """Returns length of vocabulary"""
        return len(self._vocab.keys())
