import requests
import time
from fake_useragent import UserAgent

url = "https://www.reddit.com/r/Btechtards/comments/1lkz6j6/indians_picking_career_options/"

# Generate a random user agent
session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
)
ua = UserAgent()

r = session.get(url, headers={"User-Agent": ua.random})

print(r.text)

with open("blog.html", "w", encoding="utf-8") as f:
    f.write(r.text)
