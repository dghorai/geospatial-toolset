import boto3
import io
import pickle
import numpy as np

from io import BytesIO
from urllib.parse import urlparse



s3_client = boto3.client('s3')


# bypass your local disk and upload directly the data to the cloud (AWS s3) whle creating pickle file from numpy array
my_array = np.random.randn(10)

# upload without using disk
my_array_data = io.BytesIO()
pickle.dump(my_array, my_array_data)
my_array_data.seek(0)
s3_client.upload_fileobj(my_array_data, 'your-bucket', 'your-file.pkl')

# download without using disk
my_array_data2 = io.BytesIO()
s3_client.download_fileobj('your-bucket', 'your-file.pkl', my_array_data2)
my_array_data2.seek(0)
my_array2 = pickle.load(my_array_data2)

# check that everything is correct
np.allclose(my_array, my_array2)


# create numpy array npy fle and upload to S3
def to_s3_npy(data: np.array, s3_uri: str):
    # s3_uri looks like f"s3://{BUCKET_NAME}/{KEY}"
    bytes_ = BytesIO()
    np.save(bytes_, data, allow_pickle=True)
    bytes_.seek(0)
    parsed_s3 = urlparse(s3_uri)
    s3_client.upload_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    return True


def from_s3_npy(s3_uri: str):
    bytes_ = BytesIO()
    parsed_s3 = urlparse(s3_uri)
    s3_client.download_fileobj(
        Fileobj=bytes_, Bucket=parsed_s3.netloc, Key=parsed_s3.path[1:]
    )
    bytes_.seek(0)
    return np.load(bytes_, allow_pickle=True)



# references
# https://stackoverflow.com/questions/48049557/how-to-write-npy-file-to-s3-directly