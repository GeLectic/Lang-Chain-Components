from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

search_tool = DuckDuckGoSearchRun()

results1 = search_tool.invoke('top news about Indore')

shell_tool = ShellTool()

result2 = shell_tool.invoke("ls")
