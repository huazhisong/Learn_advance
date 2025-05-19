from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
import asyncio
from requests_html import AsyncHTMLSession
import urllib3

urllib3.disable_warnings()


async def worker(s: AsyncHTMLSession, url: str):
    try:
        header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
        if "text/html" not in header_response.headers.get("Content-Type", ""):
            return None
        r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None


async def get_htmls(urls):
    session = AsyncHTMLSession()
    tasks = (worker(session, url) for url in urls)
    return await asyncio.gather(*tasks)


async def search(keyword: str, n_results: int = 3) -> List[str]:
    """
    This function will search the keyword and return the text content in the first n_results web pages.

    Warning: You may suffer from HTTP 429 errors if you search too many times in a period of time. This is unavoidable and you should take your own risk if you want to try search more results at once.
    The rate limit is not explicitly announced by Google, hence there's not much we can do except for changing the IP or wait until Google unban you (we don't know how long the penalty will last either).
    """
    keyword = keyword[:100]
    # First, search the keyword and get the results. Also, get 2 times more results in case some of them are invalid.
    results = list(_search(keyword, n_results * 2, lang="zh", unique=True))
    # Then, get the HTML from the results. Also, the helper function will filter out the non-HTML urls.
    results = await get_htmls(results)
    # Filter out the None values.
    results = [x for x in results if x is not None]
    # Parse the HTML.
    results = [BeautifulSoup(x, "html.parser") for x in results]
    # Get the text from the HTML and remove the spaces. Also, filter out the non-utf-8 encoding.
    results = [
        "".join(x.get_text().split())
        for x in results
        if detect(x.encode()).get("encoding") == "utf-8"
    ]
    # Return the first n results.
    return results[:n_results]
