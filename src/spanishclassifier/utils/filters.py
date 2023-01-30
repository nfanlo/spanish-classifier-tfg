import re


def remove_twitter_handles(row):
    tweet = row["text"]
    row["text"] = re.sub('@[^\s]+','',tweet)
    return row

def remove_urls(row):
    tweet = row["text"]
    row["text"] = re.sub(r'http\S+', '', tweet)
    return row

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def clean_html(row):
    tweet = row["text"]
    row["text"] = re.sub(CLEANR, '', tweet)
    return row