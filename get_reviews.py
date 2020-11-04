import time
import typing
import urllib
import requests
import csv
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen

#парсер для собирания датасета


def is_error_response(http_response, seconds_to_sleep: float = 1):
    if http_response.status_code == 503:
        time.sleep(seconds_to_sleep)
        return False

    return http_response.status_code != 200


def get_json(url) -> typing.Union[dict, None]:
    response = requests.get(url)
    if is_error_response(response):
        return None
    json_response = response.json()
    return json_response

def change_rating(rating):
    if rating >= '4':
        return 1
    else:
        return 0

def get_reviews(app_id, page=1):
    while True:
        url = (f'https://itunes.apple.com/ru/rss/customerreviews/id={app_id}/'
               f'page={page}/sortby=mostrecent/json')
        json = get_json(url)
        if not json:
            return 1
        if page > 10:
            return 1
        data_feed = json.get('feed')
        try:
            for entry in data_feed.get('entry'):
                if entry.get('im:name'): continue
                rating = change_rating(entry.get('im:rating').get('label'))
                review = entry.get('content').get('label')
                with open('reviews.сsv', mode='a', encoding='utf-8', newline='') as output_revs:
                    revs_writer = csv.writer(output_revs, delimiter='|', quotechar='"', quoting=csv.QUOTE_ALL)
                    revs_writer.writerow([rating, str(review)])
        except:
            pass
        page += 1

games_page = urllib.request.urlopen("https://apps.apple.com/ru/genre/ios-%D0%B8%D0%B3%D1%80%D1%8B/id6014")
soup = BeautifulSoup(games_page, features="html.parser")

#list of links(for iterating in loop)
links = []

#поиск ссылок на странице, извлечение id каждого приложения
for link in soup.findAll('a', attrs={'href': re.compile("^https://apps.apple.com/ru/app")}):
    print(link.get('href'))
    links.append(link.get('href'))
    src_string = link.get('href')
    form_string = src_string.split('/id')
    form_string = form_string[len(form_string)-1]
    print(form_string)
    get_reviews(str(form_string))