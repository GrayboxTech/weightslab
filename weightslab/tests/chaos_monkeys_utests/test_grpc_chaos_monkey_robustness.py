import os
import random
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc

from weightslab.trainer.trainer_services import (
    ExperimentServiceServicer,
    GrpcServerManager,
    RpcTimingAndWatchdogInterceptor,
    RpcWatchdogState,
)


_TEST_TIMEOUT = int(os.getenv("WL_TEST_TIMEOUT", "30"))


class _TimeoutMixin:
    """Wrap each test with a hard timeout to avoid dead CI runs."""

    def run(self, result=None):
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(super().run, result)
        try:
            fut.result(timeout=_TEST_TIMEOUT)
        except FuturesTimeoutError:
            if result is not None:
                result.addError(self, (TimeoutError, TimeoutError(
                    f"Test timed out after {_TEST_TIMEOUT}s"), None))
        finally:
            pool.shutdown(wait=False)


class TestGrpcChaosMonkeyRobustness(_TimeoutMixin, unittest.TestCase):
    """Chaos tests for gRPC watchdog/restart robustness under random stalls."""

    _ALL_GRPC_METHODS = [
        "/weightslab.ExperimentService/GetSamples",
        "/weightslab.ExperimentService/GetWeights",
        "/weightslab.ExperimentService/GetActivations",
        "/weightslab.ExperimentService/ApplyDataQuery",
        "/weightslab.ExperimentService/GetDataSamples",
        "/weightslab.ExperimentService/EditDataSample",
        "/weightslab.ExperimentService/GetDataSplits",
        "/weightslab.ExperimentService/CheckAgentHealth",
        "/weightslab.ExperimentService/GetLatestLoggerData",
        "/weightslab.ExperimentService/ExperimentCommand",
        "/weightslab.ExperimentService/ManipulateWeights",
        "/weightslab.ExperimentService/RestoreCheckpoint",
    ]

    def _build_handler(self, fate: str):
        def _rpc_handler(request, context):
            _ = request
            _ = context
            if fate == "stuck":
                # Deliberately exceed watchdog threshold to emulate a wedged worker.
                time.sleep(0.08)
            elif fate == "error":
                raise RuntimeError("chaos monkey injected handler failure")
            else:
                time.sleep(0.002)
            return {"ok": True, "fate": fate}

        return grpc.unary_unary_rpc_method_handler(
            _rpc_handler,
            request_deserializer=None,
            response_serializer=None,
        )

    def test_all_grpc_methods_survive_chaos_and_cleanup_inflight(self):
        watchdog_state = RpcWatchdogState(stuck_threshold_s=0.03)
        interceptor = RpcTimingAndWatchdogInterceptor(watchdog_state)
        rng = random.Random(20260401)
        fates = ["ok", "ok", "ok", "stuck", "error"]

        method_hits = Counter()

        def _invoke_once(method_name: str):
            fate = rng.choice(fates)
            method_hits[method_name] += 1

            def continuation(_details):
                return self._build_handler(fate)

            details = SimpleNamespace(method=method_name)
            wrapped = interceptor.intercept_service(continuation, details)

            try:
                wrapped.unary_unary(request={}, context=SimpleNamespace())
            except RuntimeError:
                # Chaos monkey is expected to kill some handlers.
                pass

        # Run each RPC many times with random outcomes so random stalls are covered.
        tasks = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            for _ in range(8):
                for method in self._ALL_GRPC_METHODS:
                    tasks.append(pool.submit(_invoke_once, method))
            for task in tasks:
                task.result(timeout=10)

        for method in self._ALL_GRPC_METHODS:
            self.assertGreater(method_hits[method], 0, f"Method never exercised: {method}")

        snap = watchdog_state.snapshot()
        self.assertEqual(snap["in_flight"], 0, "in-flight RPCs leaked after chaos run")
        self.assertFalse(snap["unhealthy"], "watchdog remained unhealthy after all requests ended")

    def test_servicer_all_grpc_commands_are_callable(self):
        exp_service = MagicMock()
        exp_service.model_service = MagicMock()
        exp_service.data_service = MagicMock()

        # Return opaque values so we can assert end-to-end call plumbing.
        exp_service.model_service.GetSamples.return_value = object()
        exp_service.model_service.GetWeights.return_value = object()
        exp_service.model_service.GetActivations.return_value = object()
        exp_service.data_service.ApplyDataQuery.return_value = object()
        exp_service.data_service.GetDataSamples.return_value = object()
        exp_service.data_service.EditDataSample.return_value = object()
        exp_service.data_service.GetDataSplits.return_value = object()
        exp_service.data_service.CheckAgentHealth.return_value = object()
        exp_service.GetLatestLoggerData.return_value = object()
        exp_service.ExperimentCommand.return_value = object()
        exp_service.model_service.ManipulateWeights.return_value = object()
        exp_service.RestoreCheckpoint.return_value = object()

        servicer = ExperimentServiceServicer(exp_service=exp_service)
        req = SimpleNamespace()
        ctx = SimpleNamespace()

        results = [
            servicer.GetSamples(req, ctx),
            servicer.GetWeights(req, ctx),
            servicer.GetActivations(req, ctx),
            servicer.ApplyDataQuery(req, ctx),
            servicer.GetDataSamples(req, ctx),
            servicer.EditDataSample(req, ctx),
            servicer.GetDataSplits(req, ctx),
            servicer.CheckAgentHealth(req, ctx),
            servicer.GetLatestLoggerData(req, ctx),
            servicer.ExperimentCommand(req, ctx),
            servicer.ManipulateWeights(req, ctx),
            servicer.RestoreCheckpoint(req, ctx),
        ]

        self.assertEqual(len(results), 12)
        self.assertTrue(all(result is not None for result in results))

    def test_watchdog_unhealthy_transition_and_recovery(self):
        watchdog_state = RpcWatchdogState(stuck_threshold_s=0.01)

        rpc_id = watchdog_state.begin("/weightslab.ExperimentService/GetSamples")
        time.sleep(0.03)
        snap_unhealthy = watchdog_state.snapshot()
        unhealthy_count = watchdog_state.record_unhealthy()

        self.assertTrue(snap_unhealthy["unhealthy"])
        self.assertEqual(unhealthy_count, 1)

        watchdog_state.end(rpc_id)
        snap_recovered = watchdog_state.snapshot()
        reset_count = watchdog_state.record_healthy()

        self.assertEqual(snap_recovered["in_flight"], 0)
        self.assertFalse(snap_recovered["unhealthy"])
        self.assertEqual(reset_count, 0)

    def _assert_stuck_restart_retry_for_method(self, method_name: str):
        watchdog_state = RpcWatchdogState(stuck_threshold_s=0.02)
        interceptor = RpcTimingAndWatchdogInterceptor(watchdog_state)
        server_manager = GrpcServerManager()
        attempt = {"n": 0}

        def continuation(_details):
            attempt["n"] += 1
            current_attempt = attempt["n"]

            def _rpc_handler(request, context):
                _ = request
                _ = context
                if current_attempt == 1:
                    # Simulate a long gRPC call that gets stuck mid-flight.
                    time.sleep(0.06)
                    if server_manager.should_restart():
                        raise RuntimeError("server restarted while request was stuck")
                return {"ok": True, "attempt": current_attempt}

            return grpc.unary_unary_rpc_method_handler(
                _rpc_handler,
                request_deserializer=None,
                response_serializer=None,
            )

        def run_one_call_with_watchdog():
            details = SimpleNamespace(method=method_name)
            wrapped = interceptor.intercept_service(continuation, details)
            result_holder = {}
            error_holder = {}

            def _invoke():
                try:
                    result_holder["value"] = wrapped.unary_unary(request={}, context=SimpleNamespace())
                except Exception as exc:  # expected on first attempt
                    error_holder["error"] = exc

            worker = threading.Thread(target=_invoke, name="WL-Test-gRPC-Worker", daemon=True)
            worker.start()

            while worker.is_alive():
                snap = watchdog_state.snapshot()
                if snap["unhealthy"]:
                    unhealthy_count = watchdog_state.record_unhealthy()
                    if unhealthy_count >= 1:
                        server_manager.request_restart()
                else:
                    watchdog_state.record_healthy()
                time.sleep(0.005)

            worker.join(timeout=1)
            return result_holder, error_holder

        first_result, first_error = run_one_call_with_watchdog()
        self.assertEqual(first_result, {})
        self.assertIn("error", first_error)
        self.assertTrue(server_manager.should_restart())

        # Emulate the server loop restart cycle before replaying the request.
        server_manager.clear_restart_request()
        self.assertFalse(server_manager.should_restart())

        second_result, second_error = run_one_call_with_watchdog()
        self.assertEqual(second_error, {})
        self.assertTrue(second_result["value"]["ok"])
        self.assertEqual(second_result["value"]["attempt"], 2)

        snap_done = watchdog_state.snapshot()
        self.assertEqual(snap_done["in_flight"], 0)
        self.assertFalse(snap_done["unhealthy"])

    def test_all_grpc_methods_stuck_then_watchdog_restart_then_retry_success(self):
        for method_name in self._ALL_GRPC_METHODS:
            with self.subTest(method=method_name):
                self._assert_stuck_restart_retry_for_method(method_name)

    def test_server_manager_restart_flag_cycle(self):
        manager = GrpcServerManager()

        self.assertFalse(manager.should_restart())
        manager.request_restart()
        self.assertTrue(manager.should_restart())
        manager.clear_restart_request()
        self.assertFalse(manager.should_restart())


if __name__ == "__main__":
    unittest.main()
