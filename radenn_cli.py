import sys
if ("radenn/" not in sys.path):
    sys.path.insert(1, "radenn/")
    
from radenn.radenn_interpreter import run

while True:
    text = input(">> ")
    if text.strip() == "":
        continue
    if text.strip().lower() == "exit":
        break
    result, error = run("<stdin>", text)
    if error:
        print(error)
    else:
        for element in result.elements:
            try:
                if element.should_print:
                    print(repr(element))
            except:
                print("There was an error")
                