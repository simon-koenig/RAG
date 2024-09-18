import pandas as pd

# Load the CSV file
file_path = "/home/simon/master_project/software/chatbot/llm_param_dev/dev/lookAtMe.csv"
df = pd.read_csv(file_path)

# Keep only the 'contexts_ids' and 'goldPassages' columns
df = df[["contexts_ids", "goldPassages"]]
print(df.head())

# Save the modified DataFrame back to CSV
df.to_csv(file_path, index=False)
