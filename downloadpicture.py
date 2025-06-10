from bing_image_downloader import downloader

downloader.download("Taylor Swift face", limit=500, output_dir='./data/raw', adult_filter_off=True, force_replace=False, timeout=60)
