import json
from tqdm import tqdm
import multiprocessing
import sys
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

def get_data(domain):
    try:
        try:
            resp = requests.get('https://{}/favicon.ico'.format(domain))
        except requests.exceptions.RequestException as e:
            try:
                resp = requests.get('http://{}/favicon.ico'.format(domain))
            except requests.exceptions.RequestException as e:
                #print('Can not request', domain)
                return None

        try:
            im = Image.open(BytesIO(resp.content))
        except OSError:
                #print('Bad icon', domain)
                return None
        if im.width != 16 or im.height != 16:
            im = im.resize((16, 16))
        im = im.convert('RGB')
        icon_data = []
        for px in im.getdata():
            try:
                icon_data.extend(px)
            except TypeError:
                #print('Bad icon', domain)
                return None

        if len(icon_data) != (16*16)*3:
            #print('Wrong len(icon_data)', len(icon_data), im, domain)
            return None

        try:
            resp = requests.get('https://{}/'.format(domain))
        except requests.exceptions.RequestException as e:
            try:
                resp = requests.get('http://{}/'.format(domain))
            except requests.exceptions.RequestException as e:
                #print('Can not request', domain)
                return None
        soup = BeautifulSoup(resp.content, 'lxml')
        raw_title = (soup.title and soup.title.string) or ''
        title = raw_title[:min(len(raw_title), 24)]

        return (domain, title, icon_data)
    except Exception as e:
        return None


if __name__ == '__main__':
    pool = multiprocessing.Pool(64)
    domains = [l.strip() for l in open('list-of-domains/top-10k.txt')]
    res = pool.imap_unordered(get_data, tqdm(domains))

    with open('data/all.jl', 'w') as f:
        for data in res:
            if data is None:
                continue
            #print(data[0])
            json.dump(data, f)
            f.write('\n')
