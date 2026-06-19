# MIT License

# Copyright (c) 2024 Sebastian Szymański

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Defines the heapq functionalities, which mimic the Python module.

Downloaded at https://raw.githubusercontent.com/sebastian-sz/heapq.mojo/refs/heads/main/heapq.mojo

```mojo
import heapq

var heap = List[Int](3, 2, 1)
heapq.heapify(heap)
for e in heap:
    print(e[])
# 1,2,3

heapq.heappush(heap, 0)
for e in heap:
    print(e[])
# 0,1,3,2

var smallest = heapq.heappop(heap)
if smallest:    # optional
    print(smallest.take())
```
"""

from std.collections import Optional


trait Comparable:
    """An element which can be greater or smaller than another element."""

    fn __lt__(self, other: Self) -> Bool:
        """Implements < (less than) operator."""
        ...

    fn __gt__(self, other: Self) -> Bool:
        """Implements > (greater than) operator."""
        ...


trait OrderedCollectionElement(
    Movable, Comparable, Copyable, ImplicitlyDestructible
):
    """An Element which can exist in a collection with some specified order."""

    pass


fn heappush[T: OrderedCollectionElement](mut array: List[T], value: T):
    """
    Pushes the element onto the heap.

    This function modifies the heap in-place to maintain the structure and
    order properties.

    Parameters:
        T: The type of elements in the list.

    Args:
        array: A reference to the List, which resembles the heap.
        value: Value to be pushed.
    """
    var pushed_value = value.copy()
    array.append(pushed_value.copy())
    var index = len(array) - 1
    var parent_index = (index - 1) // 2

    while index > 0 and pushed_value < array[parent_index]:
        array[index] = array[parent_index].copy()
        index = parent_index
        parent_index = (index - 1) // 2

    array[index] = pushed_value^


fn heappop[T: OrderedCollectionElement](mut array: List[T]) -> Optional[T]:
    """
    Removes and returns the smallest element from the heap.

    This function modifies the heap in-place to maintain the structure and
    order properties.

    Parameters:
        T: The type of elements in the list.

    Args:
        array: A reference to the List, which resembles the heap.

    Returns:
        An `Optional[T]` - if the array is empty returns None.
        Else the smallest element is removed and the array is
        updated.
    """
    if len(array) == 0:
        return Optional[T](None)

    # Swap last value with the root and pop root
    var tmp = array[0].copy()
    array[0] = array[-1].copy()
    array[-1] = tmp^
    var to_return = array.pop()

    # Rebalance the tree
    _percolate_down(array, 0)

    return to_return^


fn _percolate_down[
    T: OrderedCollectionElement
](mut array: List[T], index: Int):
    var idx = index  # Make explicit copy to prevent mutability issus

    while True:
        var left_child_idx = (2 * idx) + 1
        var right_child_idx = left_child_idx + 1

        # No left child -> no right child either -> leaf node
        if left_child_idx >= len(array):
            break

        # Find index of smaller child
        var smaller_child_idx: Int
        if (
            right_child_idx < len(array)
            and array[right_child_idx] < array[left_child_idx]
        ):
            smaller_child_idx = right_child_idx
        else:
            smaller_child_idx = left_child_idx

        if array[idx] > array[smaller_child_idx]:
            var tmp = array[idx].copy()
            array[idx] = array[smaller_child_idx].copy()
            array[smaller_child_idx] = tmp^
            idx = smaller_child_idx
        else:  # Already balanced
            break


fn heapify[T: OrderedCollectionElement](mut array: List[T]):
    """
    Transform the list into a heap, in-place, in linear time.

    This function modifies the list in-place to satisfy the heap property,
    ensuring that for every node `i` other than the root, the value of `i`
    is greater than or equal to the value of its parent.

    Parameters:
        T: The type of elements in the list.

    Args:
        array: A reference to the List, which will be transformed into a heap.
    """
    for idx in range(len(array) - 1, -1, -1):
        _percolate_down(array, idx)
