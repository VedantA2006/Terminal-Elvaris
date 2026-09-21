import os
import ast

with open('ai_research_agents.py', 'r', encoding='utf-8') as f:
    content = f.read()

tree = ast.parse(content)
node = [n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'ALPHA_ARCHETYPES'][0]
start = node.lineno
end = node.end_lineno

lines = content.split('\n')
before = lines[:start-1]
after = lines[end:]

new_code = """
def load_archetypes():
    import json
    import os
    path = os.path.join("data", "archetypes.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []

def save_archetypes(archetypes):
    import json
    import os
    path = os.path.join("data", "archetypes.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(archetypes, f, indent=4)
"""

# Replace all occurrences of ALPHA_ARCHETYPES with load_archetypes() in the after string
after_str = '\n'.join(after)
after_str = after_str.replace("len(ALPHA_ARCHETYPES)", "len(load_archetypes())")
after_str = after_str.replace("ALPHA_ARCHETYPES[", "load_archetypes()[")

with open('ai_research_agents.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(before) + '\n' + new_code + '\n' + after_str)
