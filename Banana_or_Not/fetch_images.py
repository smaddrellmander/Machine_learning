import urllib3
import shutil


url = 'http://www.solarspace.co.uk/PlanetPics/Neptune/NeptuneAlt1.jpg'
c = urllib3.PoolManager()
filename = 'save.png'

with c.request('GET',url, preload_content=False) as resp, open(filename, 'wb') as out_file:
    shutil.copyfileobj(resp, out_file)

resp.release_conn()
