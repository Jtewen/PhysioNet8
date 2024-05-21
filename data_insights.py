import os
import re
import csv
from collections import defaultdict

# Define the directory containing the files
directory_path = 'training/'

# Dictionary to store the Dx code counts per domain
dx_distribution = defaultdict(lambda: defaultdict(int))

# Regular expressions for extracting information
dx_code_regex = re.compile(r'# Dx: ([\d,]+)')

# Function to process each file
def process_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
        
        # Extract the domain from the filename (assuming the first letter is the domain identifier)
        domain = filename[0]  # Extracts the first character which is the domain identifier
        
        # Extract Dx codes
        dx_matches = dx_code_regex.search(content)
        if dx_matches:
            dx_codes = dx_matches.group(1).split(',')
            for dx in dx_codes:
                dx_distribution[domain][dx.strip()] += 1

# Process only .hea files in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.hea'):
        file_path = os.path.join(directory_path, file_name)
        process_file(file_path)

# Save the distribution of Dx codes per domain to a CSV file
with open('dx_distribution.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Domain', 'Dx Code', 'Occurrences'])
    
    for domain, dx_counts in dx_distribution.items():
        for dx, count in dx_counts.items():
            writer.writerow([domain, dx, count])

print("CSV file has been saved with Dx code distributions.")
