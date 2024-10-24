import random

print(random.choice(["heads", "tails"]))

number = random.randint(1,10)
print(number)

cards = ["jack","queen","king"]
random.shuffle(cards)
for card in cards:
    print(card)

import statistics

print(statistics.mean([100,90]))

import sys

if len(sys.argv) < 2:
    sys.exit("Too few arguments")
elif len(sys.argv) >2:
    sys.exit("Too many arguments")

print("Hello, my name is", sys.argv[1])

for arg in sys.argv[1:]:
    print("Hello, My name is", arg)

import cowsay
import sys

if len(sys.argv) == 2:
    cowsay.trex("hello, "+ sys.argv[1])

import requests
import sys
import json

if len(sys.argv) != 2:
    sys.exit()

response = requests.get(
    "https://itunes.apple.com/search?entity=song&limit=1&term="+sys.argv[1])
print(json.dumps(response.json(), indent=2))
