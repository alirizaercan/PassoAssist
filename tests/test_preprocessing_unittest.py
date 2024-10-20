import unittest
from test_preprocessing import normalize_turkish_chars, remove_unwanted_chars, preprocess_text

class TestPreprocessing(unittest.TestCase):

    def test_normalize_turkish_chars(self):
        # Testler
        self.assertEqual(normalize_turkish_chars('İstanbul', to_english=True), 'istanbul')
        self.assertEqual(normalize_turkish_chars('çalışkan', to_english=True), 'caliskan')
        self.assertEqual(normalize_turkish_chars('Çiçek', to_english=False), 'cicek')
        self.assertEqual(normalize_turkish_chars('Ğürbüz', to_english=False), 'gurbuz')

    def test_remove_unwanted_chars(self):
        # Testler
        self.assertEqual(remove_unwanted_chars("Bu bir test! @mention #hashtag http://example.com"), 
                         "Bu bir test")
        self.assertEqual(remove_unwanted_chars("Sayılar: 123, 456 ve özel karakterler: !@#$"), 
                         "Sayılar ve özel karakterler")
        self.assertEqual(remove_unwanted_chars("Tek karakter: a!"), "Tek karakter")

    def test_preprocess_text(self):
        contractions = {
            "değil": "değildir",
            "bişey": "bir şey",
            "diil": "değildir"
        }
        
        # Testler
        input_text = "benim adım Ali, çiçekler çok güzel!"
        expected_output = "benim adim ali cicekler cok guzel"
        self.assertEqual(preprocess_text(input_text, contractions, set(), to_english=True), expected_output)
        
        input_text = "bu bişey değil"
        expected_output = "bu bisey degil"
        self.assertEqual(preprocess_text(input_text, contractions, set(), to_english=True), expected_output)

if __name__ == "__main__":
    unittest.main()
