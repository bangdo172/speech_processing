import os

from os import listdir
from os.path import isfile, join
mypath = 'Tâm_sự/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
files = []
for f in onlyfiles:
	if f.find('m4a') != -1:
		ff = f.replace('m4a', 'wav')
		from_ = '{}{}'.format(mypath,f)
		from_ = from_.replace(' ', '\ ')
		to_ = '{}{}'.format(mypath, ff)
		to_ = to_.replace(' ', '_')
		files.append(ff.replace(' ', '_'))
		cmd = 'mv {} {}'.format(from_, to_)
		# cmd = cmd.replace(' ', '\ ')
		print(cmd)
		os.system(cmd)

files.sort()
print(files)
with open(mypath + 'text.txt', 'w') as file:
	for i in files:
		file.write('\n{}\n'.format(i))
# os.system()