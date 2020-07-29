def change (elements) :
    elements [0] = 888
    elements = [-3, -1, -2, -3, -4]
    print (elements[0])
numbers = [1, 4, 5]
print (numbers [0])
change (numbers)
print (numbers[0])

a = [1, 2, 3]
b = a[-2: ]
for i in b:
    i *=2
    print(i)
print ('a={}'.format(a))
print('b={}'.format(b))


a = [[1], [2], [3]]
b = a[-2: ]
for i in b:
    i *=2
    print(i)
print ('a={}'.format(a))
print('b={}'.format(b))


# 3.- 5-6 minutes, mostly to adjust the numbers

N=10
def staircase(N):
    if N<1 or N>50:
        return print('N out of limits')
    for i in range(N):
        spaces = [' ' for x in range(N - i - 1)]
        hashes = ['#' for x in range(i+1)]
        line = ''.join( spaces + hashes)
        print(line)

staircase(3)

# 4.- 10 minutes

test_str = 'abcd'
def longestseq(test_str : str):
    chars = [x for x in test_str]
    max_seq = 1
    cur_seq = 1
    for i in range(len(chars)-1):
        if chars[i+1] == chars[i] :
            cur_seq += 1
            if cur_seq > max_seq:
                max_seq = cur_seq
        else:
            cur_seq = 1
    # print(max_seq)
    return max_seq

longestseq(test_str)

from unittest import TestCase

def test_longestseq():
    _ = None
    TestCase.assertTrue(_,longestseq('aabbfffc')==3)


test_longestseq()

class Node():
    def __init__(self, value = 2, left = None, right = None):
        self.left = left
        self.right = right
        self.value = value

      def find(self, d):
        if self.data == d:
          return True
        elif d < self.data and self.left:
          return self.left.find(d)
        elif d > self.data and self.right:
          return self.right.find(d)
        return False

def ispresent(d, node: Node):
    if node.value == d:
      return True
    elif d < node.value and node.left:
      return ispresent(d, node.left)
    elif d > node.value and node.right:
      return ispresent(d, node.right)
    else:
        return False

def isPresent(value, node: Node):

ispresent(2, d)
c.r
ight.value

c= Node(5)
b= Node(1)

d = Node(3,b,c)