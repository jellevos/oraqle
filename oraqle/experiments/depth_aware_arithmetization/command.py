import random


with open("command.txt", mode="w") as file:
    print("./main " + ' '.join(f"x{i}={random.randint(5, 7)}" for i in range(35747)), file=file)
