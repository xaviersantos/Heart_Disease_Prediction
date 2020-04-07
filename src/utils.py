# prints to a text file and console
def log(file, text):
    if file:
        file.write(str(text)+"\n")
    print(text)
