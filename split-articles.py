import csv
import os
import string
import random

csv_file_path = "dutch-news-articles.csv"
output_directory = "./data"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def generate_random_string(length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        filename = generate_random_string() + '.txt'
        file_path = os.path.join(output_directory, filename)

        with open(file_path, mode='w', encoding='utf-8') as txt_file:
            txt_file.write(row['content'])

print("Files have been created successfully.")
