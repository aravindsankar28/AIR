import sys
import re
name = sys.argv[1]
with open(name) as f:
	lines = f.read().splitlines()
	for line in lines:
		pattern = re.compile('([^\s\w]|_)+')
		strippedList = pattern.sub('', line)
		print strippedList
