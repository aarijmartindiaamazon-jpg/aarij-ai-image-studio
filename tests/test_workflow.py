from src.workflow import next_chat_step


def test_chat_workflow_requests_upload_first():
    assert "Upload" in next_chat_step("Amazon image", has_product=False)


def test_chat_workflow_prioritizes_preservation():
    assert "White Background" in next_chat_step("I need an Amazon main image", has_product=True)

