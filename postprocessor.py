#!/usr/bin/env python3.7

import sys
import json

# Читаем и парсим входные данные
data = json.load(sys.stdin)

# Меняем вердикт на Manual Inspection при определенных условиях
data['verdict'] = 'MI'
data['message'] = 'Это задача, где требуется ручная проверка решения, не беспокойтесь.'

# Выводим результат
print(json.dumps(data))