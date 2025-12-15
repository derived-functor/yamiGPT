from pathlib import Path
import pytest
from yamigpt.tokenizers import BPETokenizer
from .data.tokens import tokenized_text

VOCAB_SIZE = 265

class TestBPETokenizer:
    
    @pytest.fixture(scope="session")
    def tokenizer(self) -> BPETokenizer:
        return BPETokenizer(VOCAB_SIZE)

    @pytest.fixture
    def fit_text(self) -> str:
        with open("tests/data/test_fit_text.txt", "r", encoding="utf-8") as f:
            text = f.read()

        return text

    @pytest.fixture
    def text(self) -> str:
        with open("tests/data/test_tokenize_text.txt", "r", encoding="utf-8") as f:
            text = f.read()

        return text

    @pytest.fixture
    def tokenized_text(self) -> list[int]:

        return tokenized_text

    def test_fit(self, tokenizer: BPETokenizer, fit_text: str):
        tokenizer.fit(fit_text.split("\n\n"))

        assert len(tokenizer) == VOCAB_SIZE
        assert tokenizer.merges

        # print("Merges:")
        # print(tokenizer.merges)

    def test_tokenize(self, tokenizer: BPETokenizer, text: str):
        # tokenizer.fit(text.split("\n\n"))

        print("Initial Text:")
        print(text)

        tokens = tokenizer.tokenize(text)

        assert tokens

    def test_decode(
            self,
            tokenizer: BPETokenizer,
            text: str,
            tokenized_text: list[int]
    ):
        decoded = tokenizer.decode(tokenized_text)

        assert text == decoded

    def test_save(self, tokenizer: BPETokenizer):
        path = Path("tokenizer.json")
        tokenizer.save(path)

        assert path.exists()

    def test_load(self, tokenizer: BPETokenizer):
        loaded = BPETokenizer.load("tokenizer.json")

        assert loaded._vocab_size == tokenizer._vocab_size
        assert loaded._merges == tokenizer._merges
