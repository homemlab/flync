#!/usr/bin/env python3

import aiohttp
import asyncio
from pathlib import Path

appdir = Path(__file__).resolve().parent.parent
req_links = Path(appdir) / "static" / "required_links.txt"

links = []
with open(req_links, "r") as f:
    for line in f:
        links.append(line.strip())
        
async def validate_link(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return (url, 0)
            else:
                return (url, 1)
    except Exception as e:
        return (f'Invalid URL: {url}; Error {e}', 1)

async def validate_links(links):
    async with aiohttp.ClientSession() as session:
        tasks = [validate_link(session, link) for link in links]
        results = await asyncio.gather(*tasks)
        return results

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(validate_links(links))
    for result in results:
        if result[1] == 1:
            print(f'❌ URL: {result[0]} is not reachable... exiting')
        elif result[1] == 0:
            print(f'✔ URL: {result[0]} is reachable... continuing')

    if 1 in [result[1] for result in results]:
        raise Exception("❌ One or more required links are not reachable... exiting")
    else:
        print("All required links are reachable... continuing")