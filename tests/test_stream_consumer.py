from detection_client import extract_content


def test_extract_content_from_dict():
    chunk = {"choices": [{"delta": {"content": "hello world"}}]}
    assert extract_content(chunk) == "hello world"


def test_extract_content_from_object():
    class Delta:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, delta):
            self.delta = delta

    class Chunk:
        def __init__(self, choices):
            self.choices = choices

    chunk = Chunk([Choice(Delta("streamed text"))])
    assert extract_content(chunk) == "streamed text"
