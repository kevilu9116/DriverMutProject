import sys
import os
import subprocess

if len(sys.argv) != 3:
	sys.stderr.write("The proper usage of this program is 'python transpose.py [InputMatrix] [OutputMatrix]'\n")
	sys.exit()

fn = sys.argv[1]

if fn[len(fn)-1] == "/":
	fn = fn[:len(fn) - 1]

in_file = open(fn, "r")
out_file = open(sys.argv[2], "w")

first_line = in_file.readline().strip().split(",")
my_matrix = list()

for i in range(len(first_line)):
	my_matrix.append(list())

in_file.seek(0)

for line in in_file:
	data = line.strip().split(",")
	for i in xrange(len(data)):
		my_matrix[i].append(data[i])

for line in my_matrix:
	last = len(line) - 1
	for i, x in enumerate(line):
		if i == last:
			out_file.write(x.strip())
		else:
			out_file.write(x + ",")
	out_file.write("\n")

in_file.close()
out_file.close()

