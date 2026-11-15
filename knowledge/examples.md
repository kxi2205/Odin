# ODIN Knowledge Base - Error-Fix Examples

This file contains curated error-fix examples for the ODIN RAG system.
Format: Each example has a category, error, and fix.

# Example: Pandas DataFrame append deprecated
Category: version_specific
Error:
```
AttributeError: 'DataFrame' object has no attribute 'append'
```
Fix:
```python
# Old (pandas < 2.0):
df = df.append(new_row, ignore_index=True)

# New (pandas >= 2.0):
df = pd.concat([df, new_row], ignore_index=True)
```

# Example: Requests timeout not set
Category: library_api
Error:
```
The request hangs indefinitely without returning
```
Fix:
```python
# Always set a timeout to avoid hanging forever
import requests

# Bad:
response = requests.get(url)

# Good:
response = requests.get(url, timeout=10)  # seconds
```

# Example: Pandas read_csv wrong separator
Category: library_api
Error:
```
ParserError: Error tokenizing data. C error: Expected 3 fields, saw 5
```
Fix:
```python
# Check your CSV separator/delimiter
import pandas as pd

# If your CSV uses tabs instead of commas:
df = pd.read_csv('file.csv', sep='\t')

# Or if using semicolons:
df = pd.read_csv('file.csv', sep=';')
```

# Example: itertools groupby requires sorted data
Category: uncommon_pattern
Error:
```
itertools.groupby produces unexpected groups
```
Fix:
```python
from itertools import groupby

# groupby only groups consecutive identical elements
# You must sort first if you want all identical items grouped

data = ['apple', 'banana', 'apple', 'cherry']

# Wrong (groups are split):
for key, group in groupby(data):
    print(key, list(group))
# Output: apple ['apple'], banana ['banana'], apple ['apple'], cherry ['cherry']

# Right (sort first):
for key, group in groupby(sorted(data)):
    print(key, list(group))
# Output: apple ['apple', 'apple'], banana ['banana'], cherry ['cherry']
```

# Example: asyncio run vs create_task
Category: uncommon_pattern
Error:
```
RuntimeError: asyncio.run() cannot be called from a running event loop
```
Fix:
```python
import asyncio

async def my_coroutine():
    await asyncio.sleep(1)
    return "done"

# If you're already inside an async function:
async def main():
    # Don't use asyncio.run() here - you're already in an event loop
    # Use await directly or create_task:
    result = await my_coroutine()
    # OR for concurrent execution:
    task = asyncio.create_task(my_coroutine())
    result = await task

# Only use asyncio.run() from synchronous code:
if __name__ == "__main__":
    asyncio.run(main())
```

# Example: NumPy matrix deprecated
Category: version_specific
Error:
```
PendingDeprecationWarning: the matrix subclass is not the recommended way
```
Fix:
```python
import numpy as np

# Old (deprecated since NumPy 1.20):
m = np.matrix([[1, 2], [3, 4]])

# New (use ndarray):
m = np.array([[1, 2], [3, 4]])

# Matrix multiplication:
# Old: m1 * m2
# New: m1 @ m2  or  np.matmul(m1, m2)
```

# Example: functools.reduce with empty sequence
Category: uncommon_pattern
Error:
```
TypeError: reduce() of empty sequence with no initial value
```
Fix:
```python
from functools import reduce

numbers = []

# Wrong (crashes if empty):
result = reduce(lambda x, y: x + y, numbers)

# Right (provide initial value):
result = reduce(lambda x, y: x + y, numbers, 0)
# Now returns 0 for empty list
```

# Example: Dictionary KeyError vs get
Category: library_api
Error:
```
KeyError: 'missing_key'
```
Fix:
```python
my_dict = {'a': 1, 'b': 2}

# Raises KeyError if key missing:
value = my_dict['c']  # KeyError!

# Safe alternatives:

# 1. Use get() with default:
value = my_dict.get('c', 0)  # Returns 0 if 'c' not found

# 2. Check first:
if 'c' in my_dict:
    value = my_dict['c']

# 3. Use defaultdict:
from collections import defaultdict
my_dict = defaultdict(int)  # Default value is 0
value = my_dict['c']  # Returns 0, no error
```
