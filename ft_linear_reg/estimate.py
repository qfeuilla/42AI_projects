#!/usr/bin/env python3

def deleteContent(fName):
    with open(fName, "w"):
        pass

mileage = "a"
while mileage == "a":
    inp = input("please enter the mileage of the car : ")
    try:
        mileage = int(inp)
    except:
        print("please enter an integer")

f = open("datas/model", "r+")
try:
    t0, t1 = map(float, f.read(42).split(","))
except:
    f.close()
    deleteContent("datas/model")
    f = open("datas/model", "w+")
    f.write("0,0")
    t0,t1 = [0,0]
f.close()
print("the car estimated price for {} mileage is : ".format(mileage) + str(t0 + t1 * mileage))