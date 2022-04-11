import urllib

def download_file(url, local_path):
    g = urllib.request.urlopen(url)
    with open(local_path, 'b+w') as f:
        f.write(g.read())