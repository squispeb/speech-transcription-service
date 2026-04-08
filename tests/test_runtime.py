import sys

from app.runtime import _detect_language_with_langid


class FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class FakeTensor:
    def __init__(self, values):
        self._values = values
        if values and isinstance(values[0], list):
            self.shape = (len(values), len(values[0]))
        else:
            self.shape = (len(values),)

    def __getitem__(self, index):
        value = self._values[index]
        if isinstance(value, list):
            return FakeTensor(value)
        return FakeScalar(value)


class FakeTorch:
    @staticmethod
    def softmax(logits, dim=1):
        return logits

    @staticmethod
    def topk(probabilities, k, dim=1):
        row = probabilities._values[0]
        indexed = sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:k]
        values = [[item[1] for item in indexed]]
        indices = [[item[0] for item in indexed]]
        return FakeTensor(values), FakeTensor(indices)


class FakeLangIdModel:
    def __init__(self, labels, probabilities):
        self.labels = labels
        self._probabilities = probabilities

    def infer_file(self, path):
        return None, FakeTensor([self._probabilities])


def test_detect_language_with_langid_returns_supported_label(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"test")

    decision = _detect_language_with_langid(
        langid_model=FakeLangIdModel(
            labels=["en", "es", "fr"], probabilities=[0.1, 0.81, 0.09]
        ),
        audio_path=audio_path,
        confidence_threshold=0.55,
        min_margin=0.15,
    )

    assert decision.language == "es"
    assert decision.is_decisive is True


def test_detect_language_with_langid_returns_unknown_when_ambiguous(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"test")

    decision = _detect_language_with_langid(
        langid_model=FakeLangIdModel(
            labels=["en", "es", "fr"], probabilities=[0.44, 0.41, 0.15]
        ),
        audio_path=audio_path,
        confidence_threshold=0.55,
        min_margin=0.15,
    )

    assert decision.language == "unknown"
    assert decision.is_decisive is False


def test_detect_language_with_langid_returns_unknown_for_confident_unsupported_language(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"test")

    decision = _detect_language_with_langid(
        langid_model=FakeLangIdModel(
            labels=["en", "es", "fr"], probabilities=[0.05, 0.1, 0.85]
        ),
        audio_path=audio_path,
        confidence_threshold=0.55,
        min_margin=0.15,
    )

    assert decision.language == "unknown"
    assert decision.is_decisive is True
