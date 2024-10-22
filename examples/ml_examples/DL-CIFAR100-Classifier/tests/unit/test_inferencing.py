import unittest

from CNNClassifier.pipeline.predict import PredictionPipeline


class PredictTestCase(unittest.TestCase):
    def __init__(self):
        self.pp = PredictionPipeline()

    def test_inferencing_return_zero(self):
        prediction = self.pp.predict()
        result = len(prediction)
        self.assertEqual(result, 0)

    def test_inferencing_return_type(self):
        prediction = self.pp.predict()
        result = type(prediction)
        self.assertIs(result, list)


if __name__ == '__main__':
    unittest.main()


# python -m unittest <test_file.py>
