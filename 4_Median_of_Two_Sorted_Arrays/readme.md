There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.

Example 1:

nums1 = [1, 3]
nums2 = [2]

The median is 2.0
Example 2:

nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5 
1. 先排序，把两个vector 排成一个，然后看这个vector里的元素个数是奇数还是偶数，如果是奇数，直接取排好序的新的vector的中间值，若为偶数，择取中间的两个值相加除2 
1. Binary search??? 
https://www.cnblogs.com/QG-whz/p/5194627.html 
