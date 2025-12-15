"""Abstract base classes"""

from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any

class AbstractTokenizer[TokensType](ABC, Sized):
    """Abstract base class for tokenizer"""

    @abstractmethod
    def fit(
        self,
        corpus: list[str],
        **kwargs: Any
    ) -> None:
        """Fit tokenizer to corpus of texts

        :param corpus: corpus of texts to train on.
        :param kwargs: additional params.

        :return: updates state of tokenizer.
        """

    @abstractmethod
    def tokenize(
        self,
        text: str,
        **kwargs: Any
    ) -> list[TokensType]:
        """Tokenizing text

        :param text: text to tokenize.

        :return: list of tokens (str, int or float)
        """

    @abstractmethod
    def decode(
        self,
        tokens: list[TokensType]
    ) -> str:
        """Decoding tokens back to text

        :param tokens: tokens from that specific tokenizer

        :return: initial text.
        """
