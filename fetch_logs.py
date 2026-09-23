import urllib.request, json
with urllib.request.urlopen('http://127.0.0.1:5000/api/research/status') as url:
    data = json.loads(url.read().decode())
    for inst, logs in data.get('recent_logs', {}).items():
        if logs:
            print(f'=== {inst} ===')
            for log in logs[-20:]:
                print(f"{log.get('timestamp')} [{log.get('agent')}] {log.get('message')}")
