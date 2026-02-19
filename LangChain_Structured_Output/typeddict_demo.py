from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int


new_person: Person = {'name': 'AKshar', 'age':23}