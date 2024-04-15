import requests
from bs4 import BeautifulSoup
import urllib.request
import os
import time

def fetch_images(search_term, num_pages):
    results = []
    encoded_term = urllib.parse.quote_plus(search_term)
    base_url = f"https://www.google.com/search?q={encoded_term}&tbm=isch"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    for page in range(num_pages):
        url = f"{base_url}&start={page * 20}"  
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tags = soup.find_all('img')
        
        for image in image_tags:
            src = image.get('src')
            alt = image.get('alt', 'No description available')
            if src and src.startswith('http'):
                results.append({'src': src, 'alt': alt})
        
        time.sleep(1) 

    return results

def download_images(images, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for i, image in enumerate(images):
        url = image['src']
        title = ''.join(char for char in image['alt'] if char.isalnum())
        file_name = os.path.join(save_folder, f"{title[:50]}_{i}.jpg")
        
        try:
            urllib.request.urlretrieve(url, file_name)
            print(f"Downloaded {file_name}")
        except Exception as e:
            print(f"Failed to download {url}: {str(e)}")

if __name__ == '__main__':
    search_term = "bagel"
    num_pages = 10 
    images = fetch_images(search_term, num_pages)
    download_images(images, 'scraped_images')




# import os
# import re
# import requests
# from bs4 import BeautifulSoup
# import string
# from urllib.parse import urlencode, urlparse, parse_qs



# def download_image(url, folder_path):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     try:
#         response = requests.get(url, stream=True)
#         # Clean the filename by removing invalid characters
#         file_name = url.split('/')[-1].split('?')[0]  # Take only part before '?'
#         valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
#         cleaned_file_name = ''.join(c for c in file_name if c in valid_chars)
#         file_path = os.path.join(folder_path, cleaned_file_name)
#         with open(file_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:
#                     file.write(chunk)
#         return True
#     except Exception as e:
#         print(f"Failed to save the image from {url}: {str(e)}")
#         return False

# def fetch_image_urls(query, max_images=10):
#     """
#     Fetch image URLs using Google Image Search for a given query.
#     """
#     image_urls = []
#     query_params = {'q': query, 'tbm': 'isch', 'client': 'opera', 'hl': 'en'}
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
#     url = f"https://www.google.com/search?{urlencode(query_params)}"
#     html = requests.get(url, headers=headers).text
#     soup = BeautifulSoup(html, 'html.parser')
#     for img in soup.find_all('img', {'src': re.compile('gstatic')}):
#         img_url = img['src']
#         image_urls.append(img_url)
#         if len(image_urls) >= max_images:
#             break
#     return image_urls

# def main():
#     query = "honeybees"
#     folder_path = './images/honeybees/'
#     urls = fetch_image_urls(query, max_images=10)

#     for url in urls:
#         success = download_image(url, folder_path)
#         if success:
#             print(f"Downloaded {url} successfully.")
#         else:
#             print(f"Failed to download {url}.")

# if __name__ == "__main__":
#     main()