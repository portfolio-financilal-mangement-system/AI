import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.raw.news.financial_news_data import getDataframes

# Now you can use the dataframes variable as needed
# print(getDataframes)