from app.services.ocr import TranscriptOcrService


def test_parse_text_file():
    service = TranscriptOcrService()

    result = service.parse_bytes(
        file_bytes="课程名称：高等数学\n成绩：92".encode("utf-8"),
        filename="transcript.txt",
        content_type="text/plain",
    )

    assert result.engine.startswith("text-")
    assert "高等数学" in result.text
    assert result.warning is None


def test_parse_unsupported_file():
    service = TranscriptOcrService()

    result = service.parse_bytes(
        file_bytes=b"binary",
        filename="transcript.zip",
        content_type="application/zip",
    )

    assert result.engine == "unsupported"
    assert result.text == ""
    assert result.warning is not None
