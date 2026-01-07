import re
#M.2
# MB2R
text = "M.1 M.2 M.3"
result = re.sub(r"M.(\d+)", r"MB\1R", text)

# sed -i 's/old/new/g' file.txt
#sed -E -i '' 's/M\.([0-9]+)/MB\1R/g' Interfaces/ATF2/DR_ATF2/ATF_DR_lattice.madx

print(result)



