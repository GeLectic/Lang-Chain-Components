from langchain_core.tools import tool

def multiply(a, b):
    """
    Docstring for multiply
    
    :param a: Description
    :param b: Description
    """
    return a*b

def multiply(a: int, b: int) ->int:
    """
    Docstring for multiply
    
    :param a: Description
    :type a: int
    :param b: Description
    :type b: int
    :return: Description
    :rtype: int
    """
    return a*b

@tool
def multiply(a: int, b:int) -> int:
    """
    Docstring for multiply
    
    :param a: Description
    :type a: int
    :param b: Description
    :type b: int
    :return: Description
    :rtype: int
    """

    return a*b

result = multiply.invoke({"a":5, "b":10})

print(result)
 
print(multiply.args_schema.model_json_schema())

