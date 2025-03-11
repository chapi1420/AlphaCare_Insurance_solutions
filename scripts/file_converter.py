import pandas as pd

# File paths
input_file ="/home/nahomnadew/Desktop/10x/AlphhaCare_Insurance-v2/AlphaCare_Insurance_solutions/Data/MachineLearningRating_v3.txt"
output_file ="/home/nahomnadew/Desktop/10x/AlphhaCare_Insurance-v2/AlphaCare_Insurance_solutions/Data/MachineLearningRating_v3.csv"
# Load the text file with the '|' delimiter
df = pd.read_csv(input_file, delimiter='|', low_memory=False)

# Save to CSV
df.to_csv(output_file, index=False)

print("Conversion completed. Saved to", output_file)

