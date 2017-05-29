import json
import os
import multiprocessing
import sys
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import subprocess

with open('data/all-attempted') as f:
    already_done = set([l.strip() for l in f])

def get_data(domain):
    if domain in already_done:
        return (domain, False)

    try:
        resp = requests.get('https://{}/favicon.ico'.format(domain))
    except requests.exceptions.RequestException as e:
        return (domain, 'error with getting', e)

    try:
        im = Image.open(BytesIO(resp.content))
    except (OSError, ValueError) as e:
        return (domain, 'error loading image', resp.content[:200])

    if im.width != 16 or im.height != 16:
        im = im.resize((16, 16))
    im = im.convert('RGB')

    if not os.path.isdir('data/{}'.format(domain)):
        os.mkdir('data/{}'.format(domain))
    im.save('data/{}/Y.png'.format(domain), 'png')

    subprocess.call([
        'python3', 'thumbnail.py',
        'https://{}/'.format(domain),
        'data/{}/X.png'.format(domain)])
    return (domain, True)


if __name__ == '__main__':
    pool = multiprocessing.Pool(12)
    domains = [l.strip() for l in open('list-of-domains/top-10k.txt')]
    res = pool.imap_unordered(get_data, domains)

    with open('data/all-attempted', 'a') as all_attempted_file:
        for idx, data in enumerate(res):
            print(idx, data)
            if data[1]:
                domain = data[0]
                already_done.add(domain)
                all_attempted_file.write(domain + '\n')
                all_attempted_file.flush()
