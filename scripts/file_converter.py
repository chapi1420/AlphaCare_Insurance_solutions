import pandas as pd

# File paths
input_file ="C:\\Users\\nadew\\10x\\week3\\ACIS\\data\\MachineLearningRating_v3.txt"
output_file ="C:\\Users\\nadew\\10x\\week3\\ACIS\\data\\cleaned_data\\Cleaned_Data_v4.csv"
# Load the text file with the '|' delimiter
df = pd.read_csv(input_file, delimiter='|', low_memory=False)

# Save to CSV
df.to_csv(output_file, index=False)

print("Conversion completed. Saved to", output_file)

