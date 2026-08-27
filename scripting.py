import json
with open("food_names.json") as file:
    data = json.load(file)
print(data)