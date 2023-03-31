# ----------------------------------------
# Carl's Average Calculator
#
#----------------------------------------
import math

globalcouter = 0
avg = 0
totalavg = 0

howmany = input ('Enter how many numbers we are Averaging?>> ')

while howmany > globalcouter:
    usernums = input ('What are the numbers we are averaging?>> ')
    globalcouter += 1
    avg = usernums + avg
else:
    totalavg = avg / howmany

print (totalavg)
