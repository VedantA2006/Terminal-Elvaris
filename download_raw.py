import urllib.request
import os

files = [
    "autonomous_research_loop.py",
    "ai_research_agents.py",
    "ai_generator.py"
]

base_url = "https://raw.githubusercontent.com/VedantA2006/Terminal-Elvaris/9166cf3d97042f397526453b967f54dbf1dc1137/"

for f in files:
    url = base_url + f
    print(f"Downloading {f} from {url}...")
    urllib.request.urlretrieve(url, f)
    
print("All files downloaded raw successfully!")
