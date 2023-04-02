import json
import unittest
from unittest.mock import patch

import pandas as pd

from dashboard.api import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.data = json.dumps({"feature1": [1], "feature2": [2]})

    def test_hello(self):
        response = self.app.get("/")
        self.assertEqual(response.data.decode(), "Hello World!")

    # @patch("app.pickle.load")
    # def test_predict(self, mock_load):
    #     mock_model = "fake model"
    #     mock_load.return_value = mock_model

    #     response = self.app.post("/predict", data=self.data, content_type="application/json")
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIsInstance(response.json, float)

    #     mock_load.assert_called_once()
    #     _, kwargs = mock_load.call_args
    #     self.assertIn("model.pkl", kwargs["file"].name)

    #     df = pd.read_json(self.data)
    #     mock_model.predict_proba.assert_called_once_with(df)