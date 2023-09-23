import unittest
from project2 import lucky_number

test_person = lucky_number()

class Project2Test(unittest.TestCase):
    def test_name_validation(self):
        #print("Testing boy")
        #gamerboy_test1 = lucky_number()
        #print("Testing girl")
        #gamergirl_test1 = lucky_number()
        self.assertEqual(test_person.name_validation(), "Simon Lundin")
        #self.assertEqual(gamergirl_test1.name_validation(), "Simona Lundina")

    def test_age_validation(self):
        #print("Testing age")
        #age_test1 = lucky_number()
        #self.assertEqual(age_test1.age_validation(), 20)
        self.assertEqual(test_person.age_validation(), 20)

    def test_game_start(self):
        #print("Testing game")
        #game_start_test = lucky_number()
        self.assertEqual(test_person.game_start(), "Exiting")

if __name__ == "__main__":
    unittest.main()