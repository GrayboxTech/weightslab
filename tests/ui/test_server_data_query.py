"""Tests for weightslab/ui/server.py's POST /agent-server/data-query --
lets the landing-page agent chat perform dataset/model actions (discard,
tag, sort, filter, analyze, compute stats, ...) itself, the same way the
now-retired "Backend Agent" tab's query bar always did: by calling
ExperimentService.ApplyDataQuery over the SAME upstream gRPC channel
_proxy_grpc_web already proxies everything else through.

Spins up a REAL grpc.server() implementing just ApplyDataQuery (not a mock)
alongside a real serve_ui() instance pointed at it -- same "real subprocess/
real network, not mocked" philosophy as test_server_agent.py's fake-OpenCode
HTTP server -- so this exercises the actual request-building/response-
translation code, not just its shape.
"""

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent import futures

import grpc

import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc
from weightslab.ui import server as ui_server


class _FakeExperimentService(pb2_grpc.ExperimentServiceServicer):
    """Records every request it receives and returns whatever this test set
    as `.next_response` (or raises `.next_error` instead, if set)."""

    def __init__(self):
        self.received = []
        self.next_response = pb2.DataQueryResponse(success=True, message="ok")
        self.next_error = None

    def ApplyDataQuery(self, request, context):
        self.received.append(request)
        if self.next_error is not None:
            context.abort(self.next_error[0], self.next_error[1])
        return self.next_response


class _ServerTestCase(unittest.TestCase):
    """Real serve_ui() + a real (fake) ExperimentService gRPC server behind
    it, both on ephemeral ports."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

        self.servicer = _FakeExperimentService()
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        pb2_grpc.add_ExperimentServiceServicer_to_server(self.servicer, self.grpc_server)
        backend_port = self.grpc_server.add_insecure_port("127.0.0.1:0")
        self.grpc_server.start()

        self.httpd = ui_server.serve_ui(
            ui_host="127.0.0.1", ui_port=0,
            backend_host="127.0.0.1", backend_port=backend_port,
            open_browser=False, block=False,
            experiment_dir=self.tmp,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.grpc_server.stop(grace=None)

    def _post_json(self, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method="POST", data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=10)


class TestDataQueryEndpoint(_ServerTestCase):
    def test_builds_a_real_dataqueryrequest_and_translates_the_response(self):
        self.servicer.next_response = pb2.DataQueryResponse(
            success=True, message="Discarded 3 samples.",
            number_of_all_samples=100, number_of_samples_in_the_loop=97,
            number_of_discarded_samples=3, unique_tags=["reviewed"],
            analysis_result="",
        )

        with self._post_json("/agent-server/data-query", {"query": "discard samples where loss > 5"}) as r:
            data = json.loads(r.read().decode())

        self.assertTrue(data["ok"])
        self.assertEqual(data["message"], "Discarded 3 samples.")
        self.assertEqual(data["numberOfAllSamples"], 100)
        self.assertEqual(data["numberOfSamplesInTheLoop"], 97)
        self.assertEqual(data["numberOfDiscardedSamples"], 3)
        self.assertEqual(data["uniqueTags"], ["reviewed"])

        self.assertEqual(len(self.servicer.received), 1)
        sent = self.servicer.received[0]
        self.assertEqual(sent.query, "discard samples where loss > 5")
        self.assertFalse(sent.accumulate)
        self.assertTrue(sent.is_natural_language)

    def test_accumulate_flag_is_forwarded(self):
        with self._post_json("/agent-server/data-query", {"query": "sort by loss", "accumulate": True}) as r:
            r.read()
        self.assertTrue(self.servicer.received[0].accumulate)

    def test_backend_reported_failure_is_not_an_http_error(self):
        # The backend understood the request fine and answered -- it just
        # couldn't do what was asked (ambiguous, out of scope, ...). That's
        # a normal 200 with success=false, not a transport-level failure.
        self.servicer.next_response = pb2.DataQueryResponse(
            success=False, message="I don't understand which column you mean.",
        )

        with self._post_json("/agent-server/data-query", {"query": "do the thing"}) as r:
            data = json.loads(r.read().decode())

        self.assertFalse(data["ok"])
        self.assertIn("don't understand", data["message"])
        self.assertNotIn("error", data)

    def test_empty_query_is_rejected_without_reaching_the_backend(self):
        try:
            self._post_json("/agent-server/data-query", {"query": "   "})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            data = json.loads(exc.read().decode())
        self.assertFalse(data["ok"])
        self.assertIn("required", data["error"])
        self.assertEqual(self.servicer.received, [])

    def test_grpc_failure_reports_a_clear_error_not_a_stack_trace(self):
        self.servicer.next_error = (grpc.StatusCode.INTERNAL, "dataframe not loaded yet")

        try:
            self._post_json("/agent-server/data-query", {"query": "sort by loss"})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 500)
            data = json.loads(exc.read().decode())
        self.assertFalse(data["ok"])
        self.assertIn("dataframe not loaded yet", data["error"])


if __name__ == "__main__":
    unittest.main()
