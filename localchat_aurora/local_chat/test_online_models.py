"""Online (cloud) models: key storage, request translation, SSE→NDJSON streaming, listing and error handling."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from local_chat import online_models as om
from local_chat import settings as app_settings

KEY = "sk-ant-test-0123456789abcdef"
NAME = "online:anthropic/claude-sonnet-5-5"


class Isolated(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(app_settings, "app_data_dir", lambda: Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        om._CACHE.clear()


def sse(*events):
    return io.BytesIO("".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode())


class Response(io.BytesIO):
    def __init__(self, events):
        super().__init__(sse(*events).getvalue())


STREAM = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "سلام"}},
    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " دنیا"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
    {"type": "message_stop"},
]


class Names(unittest.TestCase):
    def test_round_trip(self):
        self.assertTrue(om.is_online(NAME))
        self.assertFalse(om.is_online("qwen3:8b"))
        self.assertEqual(om.split_name(NAME), ("anthropic", "claude-sonnet-5-5"))
        self.assertEqual(om.make_name("anthropic", "x"), "online:anthropic/x")

    def test_invalid(self):
        for bad in ("online:other/x", "online:anthropic/", "qwen", None):
            with self.assertRaises(om.OnlineError):
                om.split_name(bad)


class Keys(Isolated):
    def test_save_status_delete_and_never_plain_on_status(self):
        self.assertFalse(om.key_status("anthropic")["has_key"])
        om.set_key("anthropic", f"  {KEY} ")
        status = om.key_status("anthropic")
        self.assertEqual(status, {"has_key": True, "tail": KEY[-4:]})
        self.assertNotIn(KEY, json.dumps(status))
        self.assertEqual(om.get_key("anthropic"), KEY)
        om.delete_key("anthropic")
        self.assertEqual(om.get_key("anthropic"), "")

    def test_rejects_bad_keys(self):
        for bad in ("", "short", "has space in the key 0123456789", "کلید-فارسی-" * 4, "x" * 500):
            with self.assertRaises(om.OnlineError):
                om.set_key("anthropic", bad)
        with self.assertRaises(om.OnlineError):
            om.set_key("nope", KEY)

    def test_protect_round_trip_and_garbage(self):
        self.assertEqual(om.unprotect(om.protect(KEY)), KEY)
        self.assertEqual(om.unprotect("garbage"), "")

    def test_other_settings_preserved(self):
        data = app_settings.load_settings()
        data["marker"] = 1
        app_settings._save_settings(data)
        om.set_key("anthropic", KEY)
        self.assertEqual(app_settings.load_settings().get("marker"), 1)


class BuildRequest(unittest.TestCase):
    def payload(self, messages, **options):
        return {"model": NAME, "messages": messages, "options": options}

    def test_system_roles_and_merge(self):
        _, body = om.build_request(self.payload([
            {"role": "system", "content": "S1"}, {"role": "assistant", "content": "ignored lead"},
            {"role": "user", "content": "a"}, {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"}], temperature=3, num_predict=99))
        self.assertEqual(body["model"], "claude-sonnet-5-5")
        self.assertEqual(body["system"], "S1")
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "user"])
        self.assertEqual(len(body["messages"][0]["content"]), 2)
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["max_tokens"], 99)
        self.assertTrue(body["stream"])

    def test_images_and_tool_results(self):
        png = "iVBORw0KGgoAAAANSUhEUg=="
        _, body = om.build_request(self.payload([
            {"role": "user", "content": "look", "images": [png]}, {"role": "tool", "content": "42"}]))
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["data"], png)
        self.assertIn("42", json.dumps(blocks, ensure_ascii=False))

    def test_empty_conversation_rejected(self):
        with self.assertRaises(om.OnlineError):
            om.build_request(self.payload([{"role": "system", "content": "only"}, {"role": "user", "content": "  "}]))


class Streaming(Isolated):
    def test_translates_to_ndjson_with_usage(self):
        om.set_key("anthropic", KEY)
        seen = {}

        def fake(request, timeout=0):
            seen["headers"] = {k.lower(): v for k, v in request.header_items()}
            seen["body"] = json.loads(request.data)
            return Response(STREAM)
        with patch.object(om, "urlopen", fake):
            packets = [json.loads(line) for line in om.open_stream(
                {"model": NAME, "messages": [{"role": "user", "content": "hi"}]})]
        self.assertEqual(seen["headers"]["x-api-key"], KEY)
        self.assertEqual(seen["headers"]["anthropic-version"], om.API_VERSION)
        self.assertEqual("".join(p["message"]["content"] for p in packets), "سلام دنیا")
        last = packets[-1]
        self.assertTrue(last["done"])
        self.assertEqual((last["prompt_eval_count"], last["eval_count"], last["done_reason"]), (12, 7, "stop"))

    def test_max_tokens_maps_to_length(self):
        om.set_key("anthropic", KEY)
        events = STREAM[:3] + [{"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 3}},
                               {"type": "message_stop"}]
        with patch.object(om, "urlopen", lambda r, timeout=0: Response(events)):
            last = json.loads(list(om.open_stream({"model": NAME, "messages": [{"role": "user", "content": "x"}]}))[-1])
        self.assertEqual(last["done_reason"], "length")

    def test_truncated_stream_and_error_event_raise(self):
        om.set_key("anthropic", KEY)
        for events in (STREAM[:3], STREAM[:2] + [{"type": "error", "error": {"message": "overloaded"}}]):
            with patch.object(om, "urlopen", lambda r, timeout=0, e=events: Response(e)):
                with self.assertRaises(om.OnlineError):
                    list(om.open_stream({"model": NAME, "messages": [{"role": "user", "content": "x"}]}))

    def test_missing_key(self):
        with self.assertRaises(om.OnlineError):
            om.open_stream({"model": NAME, "messages": [{"role": "user", "content": "x"}]})

    def test_http_and_network_errors(self):
        om.set_key("anthropic", KEY)
        payload = {"model": NAME, "messages": [{"role": "user", "content": "x"}]}
        err = HTTPError("u", 401, "no", {}, io.BytesIO(b'{"error":{"message":"bad key"}}'))
        with patch.object(om, "urlopen", side_effect=err):
            with self.assertRaises(om.OnlineError) as ctx:
                om.open_stream(payload)
        self.assertEqual(ctx.exception.status, 401)
        self.assertNotIn(KEY, ctx.exception.message)
        with patch.object(om, "urlopen", side_effect=URLError("down")):
            with self.assertRaises(om.OnlineError):
                om.open_stream(payload)

    def test_complete(self):
        om.set_key("anthropic", KEY)
        with patch.object(om, "urlopen", lambda r, timeout=0: Response(STREAM)):
            out = om.complete({"model": NAME, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(out["message"]["content"], "سلام دنیا")


class Listing(Isolated):
    def test_no_key_means_no_models(self):
        self.assertEqual(om.list_models()["models"], [])
        self.assertEqual(om.all_models(), [])

    def test_api_list_cached(self):
        om.set_key("anthropic", KEY)
        body = json.dumps({"data": [{"id": "claude-x", "display_name": "Claude X"}, {"bad": 1}]}).encode()
        calls = []

        def fake(request, timeout=0):
            calls.append(1)
            return io.BytesIO(body)
        with patch.object(om, "urlopen", fake):
            first = om.list_models()
            om.list_models()
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["source"], "api")
        self.assertEqual(first["models"], [{"name": "online:anthropic/claude-x", "label": "Claude X"}])

    def test_rejected_key_shows_no_models_but_network_failure_uses_default(self):
        om.set_key("anthropic", KEY)
        with patch.object(om, "urlopen", side_effect=HTTPError("u", 401, "no", {}, io.BytesIO(b"{}"))):
            result = om.list_models(refresh=True)
        self.assertEqual((result["models"], result["source"]), ([], "none"))
        self.assertTrue(result["error"])
        with patch.object(om, "urlopen", side_effect=URLError("down")):
            result = om.list_models(refresh=True)
        self.assertEqual(result["source"], "default")
        self.assertTrue(result["models"])


class ServerWiring(Isolated):
    def test_local_chat_server_handles_online_models(self):
        from local_chat import server
        self.assertTrue(hasattr(server, "online_models"))
        self.assertTrue(server._model_capabilities(NAME))
        self.assertGreater(server._model_context_capacity(NAME), 0)


if __name__ == "__main__":
    unittest.main()


class LeanOnlinePromptTests(unittest.TestCase):
    def test_greeting_to_online_model_gets_tiny_prompt(self):
        from local_chat import context_manager
        from local_chat.server import _online_profile, _system_prompt

        prompt = _system_prompt(model="online:anthropic/claude-haiku-5-5", ability_cue=False, time_cue=False,
                                identity_cue=False, web_sources=None, query="سلام")
        self.assertLess(context_manager.estimate_tokens(prompt), 150)
        for noisy in ("Ollama", "قابلیت‌های برنامه", "فارکس"):
            self.assertNotIn(noisy, prompt)
        self.assertLessEqual(_online_profile()["num_ctx"], 16384)


def oa_sse(*chunks):
    lines = "".join(f"data: {json.dumps(c)}\n\n" if c != "[DONE]" else "data: [DONE]\n\n" for c in chunks)
    return io.BytesIO((": OPENROUTER PROCESSING\n\n" + lines).encode())


class FreeProviders(Isolated):
    KEYS = {"gemini": "AIzaSyTestKey0123456789abcdef", "groq": "gsk_testkey0123456789abcdefgh",
            "openrouter": "sk-or-v1-testkey0123456789abcdef"}

    def test_names_and_limits(self):
        for provider in ("gemini", "groq", "openrouter"):
            name = om.make_name(provider, "vendor/model:free")
            self.assertEqual(om.split_name(name), (provider, "vendor/model:free"))
        self.assertLessEqual(om.limits("online:groq/llama-3.3-70b-versatile")[0], 8192)
        self.assertGreaterEqual(om.limits("online:anthropic/x")[1], 4096)

    def test_openai_request_shape(self):
        png = "iVBORw0KGgoAAAANSUhEUg=="
        provider, body = om.build_request({"model": "online:groq/llama-3.3-70b-versatile", "options": {"temperature": 0.5},
            "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "a", "images": [png]},
                         {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"}]})
        self.assertEqual(provider, "groq")
        self.assertEqual(body["model"], "llama-3.3-70b-versatile")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(body["messages"][0]["content"], "S")
        first = body["messages"][1]["content"]
        self.assertEqual(first[0]["image_url"]["url"], "data:image/png;base64," + png)
        self.assertIn("a", first[1]["text"]); self.assertIn("b", first[1]["text"])
        self.assertEqual(body["max_tokens"], om.PROVIDERS["groq"]["predict"])
        self.assertNotIn("tools", body)

    def test_streaming_and_bearer_header(self):
        om.set_key("groq", self.KEYS["groq"])
        seen = {}

        def fake(request, timeout=0):
            seen["headers"] = {k.lower(): v for k, v in request.header_items()}
            seen["url"] = request.full_url
            return oa_sse({"choices": [{"delta": {"content": "سلام"}}]},
                          {"choices": [{"delta": {"reasoning": "فکر"}}]},
                          {"choices": [{"delta": {"content": "!"}, "finish_reason": "length"}]},
                          {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}, "[DONE]")
        with patch.object(om, "urlopen", fake):
            packets = [json.loads(l) for l in om.open_stream(
                {"model": "online:groq/llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]})]
        self.assertEqual(seen["headers"]["authorization"], "Bearer " + self.KEYS["groq"])
        self.assertTrue(seen["url"].endswith("/openai/v1/chat/completions"))
        self.assertTrue(seen["headers"]["user-agent"])
        self.assertEqual("".join(p["message"]["content"] for p in packets), "سلام!")
        self.assertEqual(packets[-1]["done_reason"], "length")
        self.assertEqual((packets[-1]["prompt_eval_count"], packets[-1]["eval_count"]), (9, 4))

    def test_truncated_stream_is_an_error(self):
        om.set_key("openrouter", self.KEYS["openrouter"])
        with patch.object(om, "urlopen", lambda *a, **k: oa_sse({"choices": [{"delta": {"content": "x"}}]})):
            with self.assertRaises(om.OnlineError):
                list(om.open_stream({"model": "online:openrouter/a/b:free", "messages": [{"role": "user", "content": "hi"}]}))

    def test_model_lists_are_filtered(self):
        gemini = om._parse_models("gemini", [{"id": "models/gemini-2.5-flash", "display_name": "Flash"},
                                             {"id": "models/text-embedding-004"}, {"id": "models/gemini-live-2.5"}])
        self.assertEqual([m["name"] for m in gemini], ["online:gemini/gemini-2.5-flash"])
        groq = om._parse_models("groq", [{"id": "llama-3.1-8b-instant"}, {"id": "whisper-large-v3"}, {"id": "llama-guard-4"}])
        self.assertEqual([m["name"] for m in groq], ["online:groq/llama-3.1-8b-instant"])
        router = om._parse_models("openrouter", [
            {"id": "a/free-model:free", "name": "Free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "b/paid", "name": "Paid", "pricing": {"prompt": "0.001", "completion": "0.002"}}])
        self.assertEqual([m["name"] for m in router], ["online:openrouter/a/free-model:free"])

    def test_list_models_uses_models_endpoint_and_default_on_network_error(self):
        om.set_key("gemini", self.KEYS["gemini"])
        with patch.object(om, "urlopen", side_effect=URLError("down")):
            listing = om.list_models("gemini", refresh=True)
        self.assertEqual(listing["source"], "default")
        self.assertTrue(listing["models"][0]["name"].startswith("online:gemini/"))

    def test_vision_heuristic(self):
        self.assertTrue(om.supports_vision("online:gemini/gemini-2.5-flash"))
        self.assertFalse(om.supports_vision("online:groq/llama-3.1-8b-instant"))

    def test_error_body_formats(self):
        for raw in (b'{"error":{"message":"bad key"}}', b'[{"error":{"message":"bad key"}}]'):
            self.assertEqual(om._error_detail(raw), "bad key")
