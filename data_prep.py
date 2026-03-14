# data_prep.py
# reads the yoga-82 train file, grabs URLs from the link files, downloads images
# then makes a csv with all the labels so we can use it for feature extraction

import os
import argparse
import requests
import pandas as pd
from tqdm import tqdm
import time


# train file format is: pose_name/filename.jpg,label_6,label_20,label_82
# takes one line from yoga_train.txt (in format Akarna_Dhanurasana/64.jpg,1,8,0) 
# parses it into a dict with rel_path, pose_name, labels
def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) < 4:
        return None
    # rel_path would be Akarna_Dhanurasana/64.jpg 
    rel_path = parts[0]
    try:
        # for example: 1, 8, 0
        label_6, label_20, label_82 = int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None
    # for example: Akarna_Dhanurasana/64.jpg is Akarna_Dhanurasana
    pose_name = rel_path.split("/")[0]
    return {"rel_path": rel_path, "pose_name": pose_name,
            "label_6": label_6, "label_20": label_20, "label_82": label_82}


# each .txt file in yoga_dataset_links has lines: pose/file.jpg URL
# build a dict so we can look up the URL for each image
def build_url_index(links_dir):
    # links_dir is the folder yoga_dataset_links 
    url_index = {}
    # look through each 82 .txt files which corresponds to a pose 
    for fname in os.listdir(links_dir):
        if not fname.endswith(".txt"):
            continue
        # open the .txt file and read lines 
        with open(os.path.join(links_dir, fname), "r", errors="replace") as f:
            for line in f:
                # split by tab to get the rel_path and URL
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url_index[parts[0]] = parts[1]
    print(f"loaded {len(url_index)} URLs from link files")
    # the dict maps rel_path like "Akarna_Dhanurasana/64.jpg" to a URL string
    return url_index

# create a list of entries for each image with rel_path, pose_name, labels, url
def load_dataset_file(filepath, links_dir):
    # the dict maps rel_path like "Akarna_Dhanurasana/64.jpg" to a URL string
    url_index = build_url_index(links_dir)
    entries = []
    # filepath is yoga-82/yoga_train.txt which has lines like Akarna_Dhanurasana/64.jpg,1,8,0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # parsed is a dict with rel_path, pose_name, label_6, label_20, label_82
            parsed = parse_line(line)
            if parsed is None:
                continue
            # add the URL to the parsed dict by looking up the rel_path in url_index
            parsed["url"] = url_index.get(parsed["rel_path"])
            # add the rel_path, pose_name, label_6, label_20, label_82, url for each image to the entries list
            entries.append(parsed)
    print(f"loaded {len(entries)} entries from {filepath}")
    # a list of entries corresponding to each image 
    return entries

# download the images from the URLs, save to disk in the images folder with subfolders for each pose, for example images/Akarna_Dhanurasana/64.jpg
def download_images(entries, image_dir):
    # creates the images directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    downloaded, skipped, failed = 0, 0, 0

    # loop through the entries created with load_dataset_file, tqdm keeps track of a progress bar 
    for entry in tqdm(entries, desc="downloading"):
        if entry["url"] is None:
            failed += 1
            continue
        # build the full path to save the image, for example images/Akarna_Dhanurasana/64.jpg
        save_path = os.path.join(image_dir, entry["rel_path"])
        # creates the pose subfolder if it doesn't exist, for example images/Akarna_Dhanurasana
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # skip if path already exists (to avoid re-downloading if we run the script multiple times)
        if os.path.exists(save_path):
            skipped += 1
            continue
        # try to download the image from the URL, if try fails, it goes to except block
        try:
            # make a GET request to the URL with a timeout of 4 seconds, stream=True allows us to download large files in chunks
            # resp is the response object from requests.get, it contains the status code and the content of the response
            resp = requests.get(entry["url"], timeout=4, stream=True)
            # if the response status code is 200 (OK) 
            if resp.status_code == 200:
                with open(save_path, "wb") as f:
                    # save the content to save_path (images/Akarna_Dhanurasana/64.jpg) in chunks of 1024 bytes to avoid loading the whole file into memory
                    for chunk in resp.iter_content(1024):
                        f.write(chunk)
                downloaded += 1
            else:
                failed += 1
        # if it fails, wait 0.5s and count as failed
        except Exception:
            time.sleep(0.5)
            failed += 1

    print(f"downloaded: {downloaded}, skipped: {skipped}, failed: {failed}")

# after downloading, we validate that the files exist and are not empty, then we create a dataframe with rel_path, labels, pose_name, full_path
def validate_and_prepare(entries, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    # loop through the entries 
    for entry in entries:
        full_path = os.path.join(image_dir, entry["rel_path"])
        # check if the image file exists and is not empty, if valid, add a row to the rows list corresponding to that image 
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            rows.append({
                "rel_path": entry["rel_path"],
                "label_6": entry["label_6"],
                "label_20": entry["label_20"],
                "label_82": entry["label_82"],
                "pose_name": entry["pose_name"],
                "pose_display": entry["pose_name"].replace("_", " "),
                "full_path": os.path.abspath(full_path),
            })

    print(f"found {len(rows)}/{len(entries)} images on disk")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "yoga_prepared.csv")
    # metadata csv
    df.to_csv(csv_path, index=False)
    print(f"saved to {csv_path} ({len(df)} rows, {df['pose_name'].nunique()} poses)")
    return df


def main():
    train_file = "./yoga-82/yoga_train.txt"
    links_dir = "./yoga-82/yoga_dataset_links"
    image_dir = "./images"
    output_dir = "./train_outputs"

    # uncomment this block to run on the test set 
    """
    test_file = "./yoga-82/yoga_test.txt"
    links_dir = "./yoga-82/yoga_dataset_links"
    image_dir = "./images/test"
    output_dir = "./test_outputs"
    """

    # change train_file to test_file when running on the test set
    entries = load_dataset_file(train_file, links_dir)
    download_images(entries, image_dir)
    validate_and_prepare(entries, image_dir, output_dir)
    print("done! next run extract_features.py")


if __name__ == "__main__":
    main()