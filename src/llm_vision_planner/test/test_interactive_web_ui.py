import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from interactive_web_ui import WebUiBridge, audio_file_metadata, handler_for  # noqa: E402


class InteractiveWebUiTest(unittest.TestCase):
    def test_ui_has_mobile_audio_capture_fallback(self):
        html = (ROOT / "web" / "interactive.html").read_text(encoding="utf-8")
        self.assertIn('id="speech-file"', html)
        self.assertIn('accept="audio/webm,audio/ogg,audio/mp4', html)
        self.assertIn("speechFileInput.click()", html)

    def test_ui_displays_prediction_safety_warning(self):
        html = (ROOT / "web" / "interactive.html").read_text(encoding="utf-8")
        self.assertIn('id="safety-warning"', html)
        self.assertIn("launchProposal.safety_warning", html)
        self.assertIn("Approve & Launch Despite Warning", html)

    def test_audio_file_metadata_accepts_browser_recording_types(self):
        self.assertEqual(audio_file_metadata("audio/webm;codecs=opus"), ("audio/webm", "webm"))
        self.assertEqual(audio_file_metadata("audio/ogg; codecs=opus"), ("audio/ogg", "ogg"))
        self.assertEqual(audio_file_metadata("audio/mp4"), ("audio/mp4", "mp4"))

    def test_transcription_uses_audio_metadata_and_browser_language(self):
        request = {}

        class Transcriptions:
            @staticmethod
            def create(**options):
                request.update(options)
                return SimpleNamespace(text="  Hover near the chair.  ")

        bridge = object.__new__(WebUiBridge)
        bridge.transcription_model = "gpt-4o-mini-transcribe"
        bridge.transcription_client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=Transcriptions())
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            text = bridge.transcribe_audio(b"audio", "audio/webm;codecs=opus", "en-US")

        self.assertEqual(text, "Hover near the chair.")
        self.assertEqual(request["file"], ("operator-command.webm", b"audio", "audio/webm"))
        self.assertEqual(request["model"], "gpt-4o-mini-transcribe")
        self.assertEqual(request["language"], "en")
        self.assertEqual(request["response_format"], "json")

    def test_transcription_http_endpoint_returns_text(self):
        class Logger:
            def debug(self, message):
                del message

            def warning(self, message):
                del message

        class Bridge:
            def __init__(self):
                self.request = None

            def get_logger(self):
                return Logger()

            def transcribe_audio(self, payload, content_type, language):
                self.request = (payload, content_type, language)
                return "Hover near the chair."

        bridge = Bridge()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(bridge))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            connection.request(
                "POST",
                "/api/transcribe",
                body=b"audio",
                headers={"Content-Type": "audio/webm", "X-Speech-Language": "en-US"},
            )
            response = connection.getresponse()
            body = json.loads(response.read())
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

        self.assertEqual(response.status, 200)
        self.assertEqual(body, {"ok": True, "text": "Hover near the chair."})
        self.assertEqual(bridge.request, (b"audio", "audio/webm", "en-US"))


if __name__ == "__main__":
    unittest.main()
