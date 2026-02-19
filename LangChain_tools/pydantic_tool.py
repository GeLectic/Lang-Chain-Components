from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(json_schema_extra = True, description="the first number to add")
    b: int = Field(json_schema_extra = True, description="the second number to add")

def multiply_func(a: int, b: int) ->int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="MUltiply two numbers",
    args_schema= MultiplyInput
)

result = multiply_tool.invoke({'a':4, 'b':6})

print(result)

