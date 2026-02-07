import csv
import random
import re

# Generate 100 control commands queries
commands_queries = []
commands_status = ["Tell me the playing status.", "What's the status of playing?","status!", "Give me the status"]
commands_stop = ["Stop","Stop playing", "Stop music", "Break music", "Break playing", "Stop audio", "Cancel audio"]

for q in commands_stop:
    commands_queries.append((q, "<tool_call><name>audioplay.play_stop</name><arguments></arguments></tool_call>"))

for q in commands_status:
    commands_queries.append((q, "<tool_call><name>audioplay.play_status</name><arguments></arguments></tool_call>"))

# Write to CSV
with open('commands_queries.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['Request', 'Response'])
    for req, resp in commands_queries:
        writer.writerow([req, resp])

print("CSV file created successfully")
