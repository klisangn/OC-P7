import json
import unittest
import os
import pandas as pd
import requests

from api import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.data = json.dumps({"feature1": [1], "feature2": [2]})

    def test_hello(self):
        response = self.app.get("/")
        self.assertEqual(response.data.decode(), "Hello World!")

    def test_pred(self):
        parent_dir = os.path.abspath('..')
        file_path = os.path.join(parent_dir, "data", "sample.csv")
        data = pd.read_csv(file_path)
        x = data[data['SK_ID_CURR'] == 100028]
        # url = 'http://localhost:5000/predict'
        url = 'http://127.0.0.1:5000/predict'
        params = x.to_json()
        r = requests.post(url, json=params)
        pred = r.json()
        self.assertEqual(pred, 0.50345705959817)

        
if __name__ == '__main__':
    unittest.main()