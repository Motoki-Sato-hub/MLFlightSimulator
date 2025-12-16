import pickle

def subtract_dictionaries_by_key(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        dict_a = pickle.load(f1)
        dict_b = pickle.load(f2)
    
    if dict_a.keys() != dict_b.keys():
        print("Error: Dictionaries do not have the same keys.")
        return None

    difference_dict = {}
    for key in dict_a:
        # Check if values are numeric before trying to subtract
        if isinstance(dict_a[key], (int, float)) and isinstance(dict_b[key], (int, float)):
            difference_dict[key] = dict_a[key] - dict_b[key]
        else:
            print(f"Warning: Non-numeric value found for key {key}. Skipping subtraction for this key.")

    return difference_dict

file_path_1 = "/mnt/nas1/atf-users/userhome/pkorysko/flight-simulator-data/ATF2_Ext_20251204_234007_Dispersion/response2.pkl"
file_path_2 = "/mnt/nas1/atf-users/userhome/pkorysko/flight-simulator-data/ATF2_Ext_20251204_221116_Orbit/response2.pkl"

result_dict = subtract_dictionaries_by_key(file_path_1, file_path_2)

print(result_dict)