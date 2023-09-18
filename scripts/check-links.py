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
        return f'{url} is not valid, error: {e}'

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
            Exception(f'❌ URL: {result[0]} is not reachable... exiting')
            exit(1)
        elif result[1] == 0:
            print(f'✔ URL: {result[0]} is reachable... continuing')

    print("All required links are reachable... continuing")