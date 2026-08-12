#!/usr/bin/env python3
"""Minimal local web UI bridging operator actions to ROS 2 topics."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WebUiBridge(Node):
    def __init__(self):
        super().__init__("interactive_web_ui")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8080)
        self.declare_parameter("operator_command_topic", "/llm_vision/operator_command")
        self.declare_parameter("approval_topic", "/llm_vision/mission_approval")
        self.declare_parameter("operator_response_topic", "/llm_vision/operator_response")
        self.declare_parameter("mission_proposal_topic", "/llm_vision/mission_proposal")
        self.declare_parameter("visualizer", "standard")
        self.declare_parameter(
            "contraction_plot_path",
            "src/llm_vision_planner/plots/contraction/live_contraction.png",
        )

        self.command_pub = self.create_publisher(
            String,
            str(self.get_parameter("operator_command_topic").value),
            10,
        )
        self.approval_pub = self.create_publisher(
            String,
            str(self.get_parameter("approval_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("operator_response_topic").value),
            self.response_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mission_proposal_topic").value),
            self.proposal_callback,
            10,
        )
        self.lock = threading.Lock()
        self.latest_response = {}
        self.latest_proposal = {}
        self.contraction_plot_enabled = (
            str(self.get_parameter("visualizer").value).strip().lower() == "contraction"
        )
        self.contraction_plot_path = Path(
            str(self.get_parameter("contraction_plot_path").value)
        ).expanduser()
        self.html = self.load_html()

    @staticmethod
    def load_html():
        requested = Path(__file__).resolve().parents[1] / "web" / "interactive.html"
        candidates = [requested]
        try:
            from ament_index_python.packages import get_package_share_directory

            candidates.insert(0, Path(get_package_share_directory("llm_vision_planner")) / "web" / "interactive.html")
        except Exception:
            pass
        for candidate in candidates:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError("web/interactive.html is not installed")

    def response_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {"status": "INVALID", "message": msg.data}
        with self.lock:
            self.latest_response = payload

    def proposal_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {}
        with self.lock:
            self.latest_proposal = payload

    def state_payload(self):
        with self.lock:
            return {
                "response": self.latest_response,
                "proposal": self.latest_proposal,
                "timestamp": time.time(),
            }

    def publish_command(self, text):
        payload = {"type": "COMMAND", "text": str(text), "timestamp": time.time()}
        self.command_pub.publish(String(data=json.dumps(payload)))

    def publish_decision(self, decision, mission_id, proposal_hash):
        payload = {
            "decision": str(decision).upper(),
            "mission_id": str(mission_id),
            "proposal_hash": str(proposal_hash),
            "timestamp": time.time(),
        }
        self.approval_pub.publish(String(data=json.dumps(payload)))

    def read_contraction_plot(self):
        if not self.contraction_plot_enabled:
            return None
        try:
            payload = self.contraction_plot_path.read_bytes()
        except OSError:
            return None
        return payload if payload.startswith(b"\x89PNG\r\n\x1a\n") else None


def handler_for(bridge):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send_payload(bridge.html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self.send_json(bridge.state_payload())
                return
            if path == "/api/contraction.png":
                payload = bridge.read_contraction_plot()
                if payload is None:
                    self.send_error(404, "Contraction plot is not available")
                    return
                self.send_payload(payload, "image/png")
                return
            self.send_error(404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self.send_json({"ok": False, "error": "invalid JSON"}, status=400)
                return
            if self.path == "/api/command":
                text = str(payload.get("text", "")).strip()
                if not text:
                    self.send_json({"ok": False, "error": "command is empty"}, status=400)
                    return
                bridge.publish_command(text)
                self.send_json({"ok": True})
                return
            if self.path == "/api/decision":
                bridge.publish_decision(
                    payload.get("decision", ""),
                    payload.get("mission_id", ""),
                    payload.get("proposal_hash", ""),
                )
                self.send_json({"ok": True})
                return
            self.send_error(404)

        def send_json(self, payload, status=200):
            self.send_payload(json.dumps(payload).encode("utf-8"), "application/json", status)

        def send_payload(self, payload, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format_string, *args):
            bridge.get_logger().debug(format_string % args)

    return Handler


def main():
    rclpy.init()
    node = WebUiBridge()
    server = ThreadingHTTPServer(
        (str(node.get_parameter("host").value), int(node.get_parameter("port").value)),
        handler_for(node),
    )
    server_thread = threading.Thread(target=server.serve_forever, name="interactive_web_ui", daemon=True)
    server_thread.start()
    node.get_logger().info(f"interactive web UI available at http://{server.server_address[0]}:{server.server_address[1]}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
