import os

def restore_file(source_md, dest_py):
    with open(source_md, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The file starts with some markdown metadata, separated by '---'
    parts = content.split('---\n\n', 1)
    if len(parts) > 1:
        code = parts[1]
    else:
        code = content
        
    with open(dest_py, 'w', encoding='utf-8') as f:
        f.write(code)

if __name__ == '__main__':
    restore_file(r"C:\Users\vedan\.gemini\antigravity-ide\brain\aae4ba5f-b364-4073-8e38-98aae5c70b01\.system_generated\steps\617\content.md", "autonomous_research_loop.py")
    restore_file(r"C:\Users\vedan\.gemini\antigravity-ide\brain\aae4ba5f-b364-4073-8e38-98aae5c70b01\.system_generated\steps\618\content.md", "ai_research_agents.py")
    restore_file(r"C:\Users\vedan\.gemini\antigravity-ide\brain\aae4ba5f-b364-4073-8e38-98aae5c70b01\.system_generated\steps\619\content.md", "ai_generator.py")
    print("Restored engine files!")
