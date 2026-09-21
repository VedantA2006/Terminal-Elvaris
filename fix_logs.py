import sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open('autonomous_research_loop.py', 'r', encoding='utf-8') as f:
    text = f.read()

start_idx = text.find('def _execute_single_round')
end_idx = text.find('class AutonomousApp', start_idx)
if end_idx == -1:
    end_idx = text.find('# Global singleton', start_idx)

if start_idx != -1 and end_idx != -1:
    func_text = text[start_idx:end_idx]
    
    # Replace self._log( with _local_log(
    func_text_new = func_text.replace('self._log(', '_local_log(')
    
    # Insert _local_log definition at the beginning of the function
    insert_pos = func_text_new.find('last_round_added = False')
    if insert_pos != -1:
        local_log_def = """
        def _local_log(agent, message, level="info", instrument=instrument):
            self._log(agent, message, level, instrument=instrument)
            
        """
        func_text_new = func_text_new[:insert_pos] + local_log_def + func_text_new[insert_pos:]
    
    # Write back
    new_text = text[:start_idx] + func_text_new + text[end_idx:]
    with open('autonomous_research_loop.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print('Replaced successfully.')
else:
    print('Could not find boundaries.')
