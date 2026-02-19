from pydantic import BaseModel

class Student(BaseModel):
    name:str

new_student = {'name':"Akshar"}

student = Student(**new_student)

print(student)