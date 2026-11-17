"""Request registry: a search/other request must never evict or shadow the active chat request."""
import threading
import unittest

try:
    from . import server
except ImportError:  # pragma: no cover
    import server


class RequestRegistryTests(unittest.TestCase):
    def setUp(self):
        with server.CHAT_REQUESTS_LOCK:
            server.CHAT_REQUESTS.clear()

    def test_search_does_not_evict_active_chat_request(self):
        chat = server._begin_chat_request("sess-1", "chat-req-0001")
        search = server._begin_chat_request("sess-1", "search-req-0001", kind="search")
        self.assertIsNot(chat, search)
        # Stop for the chat request must still find it while a search is in flight.
        self.assertTrue(server._cancel_chat_request("sess-1", "chat-req-0001", wait=False))
        self.assertTrue(chat["cancelled"].is_set())
        self.assertFalse(search["cancelled"].is_set())
        server._end_chat_request(search)
        self.assertIn("sess-1", server.CHAT_REQUESTS)      # chat entry survives the search ending
        server._end_chat_request(chat)
        self.assertNotIn("sess-1", server.CHAT_REQUESTS)
        self.assertEqual(server.CHAT_REQUESTS, {})

    def test_cancel_by_id_only_hits_that_request(self):
        chat = server._begin_chat_request("sess-1", "chat-req-0001")
        search = server._begin_chat_request("sess-1", "search-req-0001", kind="search")
        self.assertTrue(server._cancel_chat_request("sess-1", "search-req-0001", wait=False))
        self.assertTrue(search["cancelled"].is_set())
        self.assertFalse(chat["cancelled"].is_set())

    def test_other_session_is_untouched(self):
        a = server._begin_chat_request("sess-a", "req-aaaaaaaa")
        b = server._begin_chat_request("sess-b", "req-bbbbbbbb")
        self.assertTrue(server._cancel_chat_request("sess-a", None, wait=False))
        self.assertTrue(a["cancelled"].is_set())
        self.assertFalse(b["cancelled"].is_set())

    def test_stale_id_does_not_cancel_new_request(self):
        old = server._begin_chat_request("sess-1", "old-req-0001")
        server._end_chat_request(old)
        new = server._begin_chat_request("sess-1", "new-req-0001")
        self.assertFalse(server._cancel_chat_request("sess-1", "old-req-0001", wait=False))
        self.assertFalse(new["cancelled"].is_set())

    def test_resources_closed_on_cancel_and_cleared_on_end(self):
        chat = server._begin_chat_request("sess-1", "chat-req-0001")
        closed = threading.Event()
        server._track_chat_resource(chat, closed.set)
        server._cancel_chat_request("sess-1", "chat-req-0001", wait=False)
        self.assertTrue(closed.is_set())
        server._end_chat_request(chat)
        self.assertEqual(chat["resources"], [])
        self.assertTrue(chat["finished"].is_set())


if __name__ == "__main__":
    unittest.main()
