#!/usr/bin/env python3.7

import sys, json

data = json.load(sys.stdin)

# Заменяем все вердикты в тестах на 'MI'
if 'tests' in data and isinstance(data['tests'], list):
    for test in data['tests']:
        if isinstance(test, dict) and 'verdict' in test:
            test['verdict'] = 'MI'

try:
    print(data['verdict'])
except Exception:
    print("no verdict field")

print(json.dumps(data))