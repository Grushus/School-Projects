import unittest
from project1 import Regressifier

class Project1test(unittest.TestCase):

    #def test_validate_data(self):
    #    csv = "C:\Python\Machine Learning\linear regression\Advertising.csv"
    #    self.assertEqual(Regressifier(csv).validate_data(),"ok")

    #def test_validate_data_regressor(self):
    #    regressor_csv = "C:\Python\Machine Learning\linear regression\Advertising.csv"
    #    self.assertEqual(Regressifier(regressor_csv).validate_data_regressor(),"ok")
    
    #def test_validate_data_classifier(self):
    #    classifier_csv = "C:\Python\Machine Learning\logistic regression\iris.csv"
    #    self.assertEqual(Regressifier(classifier_csv).validate_data_classifier(),"ok")
    
    #def test_validate_data_classifier_dummies(self):
    #    classifier_dummies_csv = "C:\Python\Machine Learning\logistic regression\penguins_size copy.csv"
    #    self.assertEqual(Regressifier(classifier_dummies_csv).validate_data_classifier(),"ok")

    #def test_regressor_ml(self):
    #    csv_regressor_ml = "C:\Python\Machine Learning\linear regression\Advertising.csv"
    #    self.assertEqual(Regressifier(csv_regressor_ml).regressor_ml(),"ok")

    def test_classifier_ml(self):
        csv_classifier_ml = "C:\Python\Machine Learning\logistic regression\iris.csv"
        self.assertEqual(Regressifier(csv_classifier_ml).classifier_ml(),"ok")


if __name__ == "__main__":
    unittest.main()