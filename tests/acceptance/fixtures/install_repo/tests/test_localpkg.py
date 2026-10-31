import unittest


class InstallTest(unittest.TestCase):
    def test_local_package_available(self):
        import localpkg

        self.assertEqual(localpkg.VALUE, "installed")
