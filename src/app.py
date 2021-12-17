import json
import numpy as np
import random
import logging
from io import BytesIO
import socketio
from PIL import Image
import glob
from sys import exit
from signal import signal, SIGINT
import tensorflow as tf
import keras
from minio import Minio
import time
from datetime import datetime
import configReader

# initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s]  [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting application...")

max_reconnection_delay_in_s = 20
# Read configuration for WebSocket client
ws_url, ws_event = configReader.get_websocket_config()
# Create WebSocket client
sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=5)

# Read configuration for object storage
obj_storage_host, bucket_url, bucket_name = configReader.get_obj_storage_config()
# Create object storage client
obj_store = Minio(
    obj_storage_host,
    access_key="admin",
    secret_key="alongerpw",
    secure=False
)

# Read configuration for model and test data
test_data_path, model_path = configReader.get_model_data_config()

def signal_handler(sig, frame):
    """
    Handle signals. Especially important for Docker

    :param sig: singal
    :type sig: int
    :param frame: Frame
    :type frame: sys.FrameType
    :return:
    """
    logger.info("Received signal %s. Gracefully shutting down application...", str(sig))
    sio.disconnect()
    logger.info("Application shutdown.")
    exit(0)


signal(SIGINT, signal_handler)


def get_reconnection_ival():
    """
    Generator to get the connection interval.
    Double the connection interval every time until the maximum interval is reached.

    :return: interval until next reconnection should be attempted
    """
    prev_ival = 0.5
    while True:
        new_ival = prev_ival * 2
        if new_ival > max_reconnection_delay_in_s:
            new_ival = max_reconnection_delay_in_s
        yield float(new_ival)
        prev_ival = new_ival


reconnect_generator = get_reconnection_ival()


@sio.event
def connect():
    """
    Callback called when a connection with the WebSocket server was successfully established

    :return:
    """
    logger.info("Client connected to WebSocket server at %s", ws_url)
    # new generator is required so retry interval will start at beginning again
    reconnect_generator = get_reconnection_ival()


@sio.event
def connect_error(data):
    """
    Callback called when there was a connection error when trying to connect to the WebSocket server.

    :param data: details on error
    :return:
    """
    logger.info("Failed to connect to WebSocket server at %s. Additional information: %s", ws_url, str(data))
    recon_attempt_in_s = reconnect_generator.__next__()
    logger.info("Attempting to reconnect in %d seconds...", int(recon_attempt_in_s))
    time.sleep(recon_attempt_in_s)
    logger.info("Trying to reconnect...")
    sio.connect(ws_url)


@sio.event
def disconnect():
    """
    Callback called when the client disconnected from the WebSocket server.

    :return:
    """
    logger.info("Client disconnected from WebSocket server at %s", ws_url)


logger.info("Connecting to Websocket server")
sio.connect(ws_url)


def convert_to_png_bytes(image):
    """
    Convert an image from its numpy array representation to a PNG byte stream.
    :param image: Image
    :type image: np.array
    :return:
    """
    img_bytes_arr = BytesIO()
    # convert nparray back to pillow image
    im = convert_to_png(image)
    # save image to byte stream
    im.save(img_bytes_arr, format="PNG")
    # move file pointer back to the start
    img_bytes_arr.seek(0)
    return img_bytes_arr

def convert_to_png(image):
    """
    Convert an image from its numpy array representation to a PNG.

    :param image: Image
    :type image: np.array
    :return:
    """
    dmin = np.min(image)
    dmax = np.max(image)
    drange =dmax-dmin
    dscale = 255.0/drange
    
    return Image.fromarray(((image-dmin)*dscale).astype('uint8'), 'RGB')


def upload_img_to_object_storage(fname, image, metadata):
    """
    Upload an image to S3 compatible object storage.

    :param fname: filename of the image
    :type fname: str
    :param image: data for the image
    :type image: np.array
    :param metadata: metadata for uploaded image
    :type metadata: dict of object
    :return: object information
    """
    img_png_bytes = convert_to_png_bytes(image)
    logger.debug("Uploading image %s to %s/%s/%s...", fname, bucket_url, bucket_name, fname)
    return obj_store.put_object(bucket_name, fname, img_png_bytes, img_png_bytes.getbuffer().nbytes,
                                metadata=metadata)


def detect_outlier(img, autoencoder):
    """
    Check if an image is an outlier or not.

    :param img: image to check
    :type img: np.array
    :param autoencoder: autoencoder
    :type autoencoder: keras model instance
    :return: True if image is an outlier, false otherwise and quadratic error
    """
    reco = autoencoder.predict(img[None])[0]
    msqe = np.average(tf.square(reco - img))
    if (msqe > 0.01):
        return True, msqe
    else:
        return False, msqe


def random_img_to_np(path, resize=True):
    """
    Get a random picture from the specified directory.

    :param path: path to directory with images
    :type path: str
    :param resize: Resize picture to 128x128?
    :type resize: bool
    :return: image as np.array
    """
    fpaths = glob.glob(path, recursive=True)
    random.shuffle(fpaths)
    logger.debug(f"Using image {fpaths[0]}")
    img = Image.open(fpaths[0]).convert("RGB")
    if (resize): img = img.resize((128, 128))
    return np.asarray(img).astype('float32') / 255

logger.debug("Loading model...")
autoencoder = keras.models.load_model(model_path)
image_chosen_ival = 5
while True:
    # choose a random image
    img = random_img_to_np(test_data_path)
    # run image through model
    img_rec = autoencoder.predict(img[None])[0].reshape(128, 128, 3)
    # check for outlier
    is_outlier, msqe = detect_outlier(img, autoencoder)
    # common values
    utc_now = datetime.utcnow()
    utc_4_fname = utc_now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    metadata = {"timestamp": utc_now}

    # upload original image
    img_fname = f"original_128x128_{utc_4_fname}.png"
    upload_img_to_object_storage(img_fname, img, metadata=metadata)

    # upload reconstructed image
    img_rec_fname = f"reconstructed_128x128_{utc_4_fname}.png"
    upload_img_to_object_storage(img_rec_fname, img_rec, metadata=metadata)

    dataStr = json.dumps({
        "time": utc_now.isoformat(),
        "images": {
            "original": f"{bucket_url}/{img_fname}",
            "reconstructed": f"{bucket_url}/{img_rec_fname}",
        },
        "details": {
            "isOk": not is_outlier,
            "quadraticError": str(msqe)
        }
    })
    logger.debug("Emitting to event %s: %s", ws_event, dataStr)
    sio.emit(ws_event, dataStr)

    time.sleep(image_chosen_ival)
