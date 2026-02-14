class Node:
  def __init__(self, value, next=None):
    self.value= value
    self.next=next
    # self.prev=prev

  def __str__(self):
    return str(self.value)


Head = Node(1)
A = Node(2)
B = Node(3)
C = Node(4)

Head.next= A
A.next = B
B.next = C

def search(head, value):
  curr = head
  while curr:
    if value == head.value:
      return True
    else:
      curr = curr.next

#find middle
def middle(head):
  slow = head
  fast = head

  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

  return slow

print(middle(Head))  


def inset

def display(head):
  element = []
  while head:
    element.append(str(head.value))
    head = head.next
  print(" -> ".join(element))

display(Head)

