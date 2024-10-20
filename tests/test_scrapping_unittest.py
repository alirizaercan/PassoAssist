import unittest
from test_scrapping import scrape_test

class TestScraping(unittest.TestCase):

    def test_scrape_test(self):
        complaints = scrape_test()
        # Şikayetlerin boş olmadığını kontrol et
        self.assertIsNotNone(complaints)
        self.assertGreater(len(complaints), 0)
        self.assertIn('text', complaints[0])  # Şikayetlerin 'text' alanı içerip içermediğini kontrol et

if __name__ == '__main__':
    unittest.main()
