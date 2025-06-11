import os
import csv
import asyncio
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from twikit import Client

# Load constants and environment variables
load_dotenv()
NUM_ACCOUNTS = int(os.getenv("NUM_ACCOUNTS", 3))
TARGET_TWEETS = int(os.getenv("TARGET_TWEETS", 60))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 20))
DELAY_MIN = float(os.getenv("DELAY_MIN", 60))
DELAY_MAX = float(os.getenv("DELAY_MAX", 80))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "scraped_data")
COOKIE_DIR = os.getenv("COOKIE_DIR", "cookies")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COOKIE_DIR, exist_ok=True)

# Load account credentials
def load_accounts(n):
    accounts = []
    for i in range(1, n + 1):
        user = os.getenv(f"TWITTER_USERNAME_{i}")
        email = os.getenv(f"TWITTER_EMAIL_{i}")
        pwd = os.getenv(f"TWITTER_PASSWORD_{i}")
        if not all([user, email, pwd]):
            raise RuntimeError(f"Missing credentials for account {i}")
        accounts.append({
            "id": i,
            "username": user,
            "email": email,
            "password": pwd,
            "cookie": os.path.join(COOKIE_DIR, f"cookies_{user}.json")
        })
    return accounts

async def authenticate(acc):
    client = Client('en-US')
    if os.path.exists(acc['cookie']):
        client.load_cookies(acc['cookie'])
        user = await client.user()
        if getattr(user, 'screen_name', None):
            return client
    await client.login(acc['username'], acc['email'], acc['password'])
    client.save_cookies(acc['cookie'])
    return client

def serialize(tweet):
    u = tweet.user
    data = {k: getattr(tweet, k, None) for k in ['id', 'created_at', 'url', 'text', 'lang']}
    data.update({
        'user_id': u.id,
        'username': u.screen_name,
        'display_name': u.name,
        'hashtags': ", ".join(tag['text'] for tag in tweet.entities.get('hashtags', [])),
        'reply_count': tweet.reply_count,
        'retweet_count': tweet.retweet_count,
        'like_count': tweet.favorite_count,
        'quote_count': tweet.quote_count,
        'view_count': tweet.view_count,
        'source': tweet.source,
        'searched_keyword': tweet.searched_keyword,
        'scraped_by': tweet.scraped_by
    })
    return data

async def scrape(keyword, start_date, end_date):
    accounts = load_accounts(NUM_ACCOUNTS)
    clients = [await authenticate(acc) for acc in accounts]
    results = []
    day = start_date

    while day < end_date:
        next_day = day + timedelta(days=1)
        q = f"{keyword} since:{day:%Y-%m-%d} until:{next_day:%Y-%m-%d}"
        total = 0
        idx = 0
        while total < TARGET_TWEETS:
            client = clients[idx % len(clients)]
            tweets = await client.search_tweet(q, 'Latest', count=BATCH_SIZE)
            if not tweets:
                break
            for t in tweets:
                if total >= TARGET_TWEETS:
                    break
                t.searched_keyword = q
                t.scraped_by = accounts[idx % len(accounts)]['id']
                results.append(serialize(t))
                total += 1
            idx += 1
            await asyncio.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
        day = next_day

    # Save to CSV
    if results:
        fname = f"tweets_{keyword}_{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}.csv"
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} tweets to {path}")
    else:
        print("No tweets found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Async Twitter Scraper")
    parser.add_argument('keyword')
    parser.add_argument('start', type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    parser.add_argument('end',   type=lambda s: datetime.strptime(s, '%Y-%m-%d'))
    args = parser.parse_args()

    asyncio.run(scrape(args.keyword, args.start, args.end))
