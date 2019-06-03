---
title: "LeetCode python 【持续更新】"
date: 2019-05-12
draft: false
tags: ["LeetCode"]
categories: ["算法"]
author: "xiaoxie"

autoCollapseToc: true
contentCopyright: '<a href="https://github.com/gohugoio/hugoBasicExample" rel="noopener" target="_blank">See origin</a>'

---
# 【写在前面】

最近在用python刷[LeetCode](https://leetcode.com/problemset/all/)，一边刷题一边对不同类型的题目进行总结。
方便以后自己翻看，同时也可以给大家提供参考。欢迎感兴趣的小伙伴一起交流哈@——@。

# 贪婪算法 greeding

## 集合覆盖问题


```python
states_need = set(['mt','wa','or','id','nv'])
stations = {}
stations['kone'] = set(['id','nv','or'])
stations['ktwo'] = set(['wa','id','mt'])
stations['kthree'] = set(['nv','or'])
final_station = set()

while(states_need):    
    best_station = None
    station_covered = set()
    for station, states_for_station in stations.items():
        coverd = states_for_station & states_need
        if(len(coverd)>len(station_covered)):
            best_station = station
            station_covered = coverd
    stations.pop(best_station)
    final_station.add(best_station)
    states_need -= station_covered

print(final_station)
```

## Best Time to Buy and Sell Stock II

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:
Input: [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
             Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.

Example 2:
Input: [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
             Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are
             engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.


```python
class Solution:
# 代码调试请加
# import pysnooper
#     @pysnooper.snoop()
#     @pysnooper.snoop('/my/log/file.log')  ### 重定向
    def maxProfit(self, prices: list) -> int:
        if(len(prices)<2): return 0
        profit,index = 0,1
        for i in range(1,len(prices)):
            if(prices[i]<prices[i-1]):
                index = i
                #break
            elif(prices[i]==prices[i-1]):
                index = i
            else:
                index = i
                profit += prices[i]-prices[i-1]
                break
#         print(index)
        return profit+self.maxProfit(prices[index:])
    
prices = [7,1,5,3,6,4]
Solution().maxProfit(prices)

```

# 动态规划 DP

## Longest Palindromic Substring

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
    
Example 2:

Input: "cbbd"
Output: "bb"


```python
class Solution:
#     @pysnooper.snoop()
    def longestPalindrome(self, s: str) -> str:
        lenth = len(s)
        maxr = 0; index = 0
        dp = [[0]*lenth for i in range(lenth)]
        r = s[::-1]
        for i in range(lenth):
            for j in range(lenth):
                if(s[i]==r[j]):
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = 0
                if(dp[i][j]>maxr): 
                    maxr = dp[i][j]
                    index = i
        print(dp)
        return maxr,index,s[index+1-maxr:index+1]
                    
#     @pysnooper.snoop()
    def longestPalindrome2(self, s):
        st = 0
        maxl = 0
        lenth = len(s)
        dp = [[False]*lenth for i in range(lenth)]
        for i in reversed(range(len(s))):
            for j in range(i,len(s)):
                if s[i] == s[j] and (i+1>j-1 or dp[i+1][j-1] ):
                    dp[i][j] = True
                    if j-i+1 > maxl:
                        st = i
                        maxl = j-i+1

        return s[st:st+maxl]

    def longestPalindrome3(self, s):
        n = len(s)
        dp = [[0]*n for i in range(n)]
        for i in range(n - 1,-1,-1):
            dp[i][i] = 1
            for j in range(i + 1,n):
                print(i,j)
                if (s[i] == s[j]):
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                else:
                    print(i,j,dp[i + 1][j], dp[i][j - 1])
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        print(dp)
        return dp[0][n - 1]    
    
    def longestPalindrome4(self, s):
        st = 0
        maxl = 0
        dp = [[False]*len(s) for i in range(len(s))]
        for i in range(len(s)-1,-1,-1):
            for j in range(i,len(s)):
                if(s[i]==s[j] and (j-i<2 or dp[i+1][j-1])):
                    dp[i][j] = True
                    if(j-i+1>maxl):
                        st = i
                        maxl = j-i+1
        return s[st:st+maxl]
        
s = 'aab'
Solution().longestPalindrome4(s)
```

## Best Time to Buy and Sell Stock

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
             
Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.


```python
class Solution:
    def maxProfit(self, prices: list) -> int:
        lenth = len(prices)
        if(lenth<2): return 0

        dp = [0]*lenth
        for i in range(1,lenth):
            dp[i] = max(dp[i-1]+prices[i]-prices[i-1],0)
        return max(dp)
    
prices = [7,1,5,3,6,4]
Solution().maxProfit(prices)

```

## Range Sum Query - Immutable

Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1

sumRange(2, 5) -> -1

sumRange(0, 5) -> -3

Note:
You may assume that the array does not change.
There are many calls to sumRange function.


```python
class NumArray:

    def __init__(self, nums: list):
        self.sum = [0] * (len(nums)+1)
        for i in range(0,len(nums)):
            self.sum[i+1] = self.sum[i]+nums[i]
    def sumRange(self, i: int, j: int) -> int:
        return self.sum[j+1]-self.sum[i]


# Your NumArray object will be instantiated and called as such:
nums, i, j = [-2, 0, 3, -5, 2, -1],0,2
obj = NumArray(nums)
obj.sumRange(i,j)
```

## Unique Paths

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
![image.png](attachment:image.png)

Above is a 7 x 3 grid. How many possible unique paths are there?

Note: m and n will be at most 100.

Example 1:

Input: m = 3, n = 2
Output: 3
    
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right

Example 2:

Input: m = 7, n = 3
Output: 28


```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n for i in range(m)]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[m-1][n-1]
m,n = 3,3
Solution().uniquePaths(m,n)
```

## Unique Paths II

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?
![image.png](attachment:image.png)
An obstacle and empty space is marked as 1 and 0 respectively in the grid.

Note: m and n will be at most 100.

Example 1:

Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2

Explanation:
There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right


```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: list) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0]*n for i in range(m)]
        for i in range(0,m):
            if(obstacleGrid[i][0]==0):
                dp[i][0] = 1
            else:    
                break
        for j in range(0,n):
            if(obstacleGrid[0][j]==0):
                dp[0][j] = 1
            else:
                break
                                
        for i in range(1,m):
            for j in range(1,n):
                if(obstacleGrid[i][j]==1): 
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[m-1][n-1]
    
Grid =  [ [0,0,0], [0,1,0], [0,0,0] ]
Solution().uniquePathsWithObstacles(Grid)        
```

## Minimum Path Sum

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:

Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
    
Explanation: Because the path 1→3→1→1→1 minimizes the sum.


```python
class Solution:
    def minPathSum(self, grid: list) -> int:
        r = len(grid)
        c = len(grid[0])
        dp = [[0] * c]
        dp[0] = grid[0][0]
        for i in range(1,c):
            dp[i] = dp[i-1] + 1
        for j in range(1,r):
            dp[0][j] = dp[0][j-1] += 1
        for i in range(1,c):
            for j in range(1,r):
                dp[i][j]
        return r,c
    
grid = [ [1,3,1], [1,5,1], [4,2,1] ]
Solution().minPathSum(grid)
```

# 华为编程题

## [编程题] 汽水瓶
时间限制：1秒

空间限制：32768K

有这样一道智力题：“某商店规定：三个空汽水瓶可以换一瓶汽水。小张手上有十个空汽水瓶，她最多可以换多少瓶汽水喝？”答案是5瓶，方法如下：先用9个空瓶子换3瓶汽水，喝掉3瓶满的，喝完以后4个空瓶子，用3个再换一瓶，喝掉这瓶满的，这时候剩2个空瓶子。然后你让老板先借给你一瓶汽水，喝掉这瓶满的，喝完以后用3个空瓶子换一瓶满的还给老板。如果小张手上有n个空汽水瓶，最多可以换多少瓶汽水喝？ 

输入描述:
输入文件最多包含10组测试数据，每个数据占一行，仅包含一个正整数n（1<=n<=100），表示小张手上的空汽水瓶数。n=0表示输入结束，你的程序不应当处理这一行。


输出描述:
对于每组测试数据，输出一行，表示最多可以喝的汽水瓶数。如果一瓶也喝不到，输出0。


输入例子1:
3
10
81
0

输出例子1:
1
5
40


```python
def bottle(n):
    res = 0
    while(n>1):
        if(n == 2): 
            res+=1
            break
        res += n//3
        n = n//3 + n%3
    return res
        
def main():
    f = open("input.txt")
    lines = f.readline()
    while(int(lines)!=0):
        print(bottle(int(lines)))
        lines = f.readline()

if __name__ == '__main__':
    main()
    
```

## [编程题] 明明的随机数
时间限制：1秒

空间限制：32768K

明明想在学校中请一些同学一起做一项问卷调查，为了实验的客观性，他先用计算机生成了N个1到1000之间的随机整数（N≤1000），对于其中重复的数字，只保留一个，把其余相同的数去掉，不同的数对应着不同的学生的学号。然后再把这些数从小到大排序，按照排好的顺序去找同学做调查。请你协助明明完成“去重”与“排序”的工作(同一个测试用例里可能会有多组数据，希望大家能正确处理)。



Input Param

n               输入随机数的个数

inputArray      n个随机整数组成的数组


Return Value

OutputArray    输出处理后的随机整数



注：测试用例保证输入参数的正确性，答题者无需验证。测试用例不止一组。




输入描述:
输入多行，先输入随机整数的个数，再输入相应个数的整数


输出描述:
返回多行，处理后的结果


输入例子1:
11
10
20
40
32
67
40
20
89
300
400
15

输出例子1:
10
15
20
32
40
67
89
300
400


```python
import sys
arr,num = [], 0
# for line in sys.stdin:

def opt(arr):
    arr = list(set(arr))
    arr.sort()
    for item in arr:
        print(item)
    
f = open("input.txt")
line = f.readline()
while(line):  
    if(num):
        arr.append(int(line))
        num -= 1
    else:
        if(arr):
            opt(arr)
        arr,num = [],int(line)
    line = f.readline()
if(arr): opt(arr)
```

## [编程题] 进制转换
时间限制：1秒

空间限制：32768K

写出一个程序，接受一个十六进制的数值字符串，输出该数值的十进制字符串。（多组同时输入 ）


输入描述:
输入一个十六进制的数值字符串。


输出描述:
输出该数值的十进制字符串。


输入例子1:
0xA

输出例子1:
10


```python
def conv16to10(s):
    if(not s): return
    flag, res = 1, 0
    ref = {'A':10,'B':11,'C':12,'D':13,'E':14,'F':15}
    for i in range(len(s)-1,-1,-1):
        if(s[i] == 'x'):
            break
        elif(s[i].isalpha()):
            res += ref[s[i]]*flag
        else:
            res += int(s[i])*flag
        flag *= 16
    return res

conv16to10('0xC460')
```


