# name = input("what's your name? ").strip().title()
# first, last = name.split(" ")
# print(f"Hello, {first}")


# x = float(input("What's x? "))
# y= float(input("What's y? "))

# z = (x/y)

# print(f"{z:.2f}")

# def main():
#     # hello(input("What's your name? "))
#     x = int(input("What is x? "))
#     print("x squared is : ", square(x))
    
# def hello(to="World"):
#     print("Hello,", to)

# def square(a):
#     return a*a

# main()

# i =3 

# while i!=0:
#     print("hello")
#     i-=1

# for _ in range(3):
#     print("haha")

# print("meow\n"*3, end="")

# while True:
#     n = int(input("What's n? "))
#     if n>0: 
#         break

# for _ in range(n):
#     print("meow")


# students = ["Harry", "Hermione", "Ron"]
# print(students)

# students = {
#     "Hermione": "Gryffindor",
#     "Harry":"Gryffindor",
#     "Ron":"Gryffindor",
#     "Draco":"Slytherin"
# }

# for s in students:
#     print(s, students[s], sep=", ")

# students= [
#     {"name":"Hermione","house":"Gryffindor","patronus":"Otter"},
#     {"name":"Harry","house":"Gryffindor","patronus":"Stag"},
#     {"name":"Ron","house":"Gryffindor","patronus":"Jack Russell Terrier"},
#     {"name":"Draco","house":"Slytherin","patronus":None}
# ]

# for student in students:
#     print(student["name"],student["house"],student["patronus"], sep=", ")

# def main():
#     print_square(3)

# def print_square(n):
#     for _ in range(n):
#         print("#"*n)

# main()
def main():
    x = get_int()
    print(f"x is {x}")

def get_int(): 
    while True:
        try: 
            return int(input("What's x? "))
        except ValueError:
            pass

main() 