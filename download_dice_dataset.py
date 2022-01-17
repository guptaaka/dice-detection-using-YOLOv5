"""
Download the dice detection dataset from Google drive links. This dataset has been
gathered from Roboflow and has been exploded by using the similar logic as present
in the _multiply_dataset.py_ script. After exploding the dataset into a huge number
of files, it is divided into training, validation and testing in a 70:20:10 ratio.
Each google drive link used below has two directories: images and labels which
contain the corresponding jpg and txt files.
"""
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_data():
    gdd.download_file_from_google_drive(file_id='1FVlJaZYNXEBwwJOiiTzCrroriu85zrNo',
                                        dest_path='./train.zip',
                                        unzip=True)
    gdd.download_file_from_google_drive(file_id='1T1ZkN2lL0MjEY9rkQz7MGwEADktMcAcd',
                                        dest_path='./valid.zip',
                                        unzip=True)
    gdd.download_file_from_google_drive(file_id='1an5pQD3mPrdKAWQ3U9YsR5M5JttoN6Ay',
                                        dest_path='./test.zip',
                                        unzip=True)

def main():
    download_data()

if __name__ == '__main__':
    main()

