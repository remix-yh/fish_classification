import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os

subscription_key = "8e87838d2ae04cd6883484b5b6b0f408"
assert subscription_key

search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
directory_name = "kasago"
search_term = "カサゴ"

thumbnail_urls = []

for i in range(4):
        headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
        params  = {"q": search_term, "license": "All", "imageType": "photo", "count": "50", "offset":str(i*50)}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        thumbnail_urls.extend([img["thumbnailUrl"] for img in search_results["value"][:]][:])

for i in range(len(thumbnail_urls)):
        image_data = requests.get(thumbnail_urls[i])
        image_data.raise_for_status()
        image = Image.open(BytesIO(image_data.content))
        if not os.path.exists("./image_source/"):
            os.makedirs("./image_source/", exist_ok=True)
        if not os.path.exists(os.path.join("./image_source/",directory_name)):
            os.makedirs(os.path.join("./image_source/",directory_name), exist_ok=True)
        image.save(os.path.join("./image_source/",directory_name, (str(i) +".bmp")))