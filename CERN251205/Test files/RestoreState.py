from Interfaces.ATF2.InterfaceATF2_Linac import InterfaceATF2_Linac
from State import State
import os

# Get the list of all files in the current directory
files = [f for f in os.listdir('..') if os.path.isfile(f) and f.startswith("machine_status")]

# Reset
filename = files[0]
print(f'Resetting the machine to file {filename}...')
I = InterfaceATF2_Linac()
S = State()
S.load(files[0])
S.push(I)

print('Done!')
