import unittest

import app


class AppTest(unittest.TestCase):
    def test_value_is_available(self):
        self.assertIn(app.VALUE, {"before", "after"})
