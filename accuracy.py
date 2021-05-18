import sys

lines1 = [' '.join(line.split()) for line in open(sys.argv[1])]
lines2 = [' '.join(line.split()) for line in open(sys.argv[2])]

if len(lines1) != len(lines2):
    raise ValueError()

total = correct = 0
for line1, line2 in zip(lines1, lines2):
    total += 1
    if line1 == line2:
        correct += 1
    elif False:
        print(line1)
        print(line2)
        print('--')

print(correct/total)

