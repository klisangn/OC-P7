import json
import unittest
from unittest.mock import patch
import os
import pandas as pd

from api import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.data = json.dumps({"feature1": [1], "feature2": [2]})

    def test_hello(self):
        response = self.app.get("/")
        self.assertEqual(response.data.decode(), "Hello World!")

if __name__ == '__main__':
    unittest.main()