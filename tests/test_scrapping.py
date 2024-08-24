import sys
import os
import unittest

# Add the 'scripts' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from scraping import scrape_page

class TestScraping(unittest.TestCase):
    def test_scrape_first_page(self):
        # URL of the first page
        page_url = 'https://www.sikayetvar.com/passo?page=1'
        
        # Scrape data from the first page
        complaints = scrape_page(page_url)
        
        # Print the results for verification
        print("Scraped complaints from the first page:")
        for complaint in complaints:
            print(complaint)
        
        # Check if complaints are retrieved
        self.assertGreater(len(complaints), 0, "No complaints found on the page")

if __name__ == "__main__":
    unittest.main()
