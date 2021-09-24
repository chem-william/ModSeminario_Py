import os

directories = os.listdir('../Python_Testing_Final_Small_Structure/')

print(directories)

for d in directories:
    print(d)
    s = "python modified_Seminario_method.py '../Python_Testing_Final_Small_Structure/" + d + "/' '../Python_Testing_Final_Small_Structure/" + d + "/' 0.957"
    print(s)
    os.system(s)
