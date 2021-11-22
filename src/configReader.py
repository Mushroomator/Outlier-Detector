import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s]  [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_websocket_config():
    """
    Get configuration for websocket from environment variables.

    :return: URL for websocket and event data is sent to
    """
    domain = os.getenv("WEBSOCKET_DOMAIN", "localhost")  #
    port = os.getenv("WEBSOCKET_PORT", 5000)
    url = f"ws://{domain}:{str(port)}"
    event = os.getenv("WEBSOCKET_EVENT", "/outlier-detection/result")
    logger.info(
        """
        Websocket configuration:
            - Domain: \t%s
            - Port: \t%s
            - URL: \t\t%s
            - Event: \t%s
        """, domain, str(port), url, event)
    return url, event


def get_obj_storage_config():
    """
    Get configuration for object storage from environment variables.

    :return: hostname (= protocol, domain, port), URL for bucket, name of bucket
    """
    domain = os.getenv("OBJ_STORE_DOMAIN", "localhost")
    port = os.getenv("OBJ_STORE_API_PORT", 9000)
    host = f"{domain}:{str(port)}"
    name = os.getenv("OBJ_STORE_BUCKET", "model-images")
    url = f"http://localhost:{str(port)}/{name}"
    logger.info(
        """
        Object storage configuration:
            - Domain: \t\t%s
            - Port: \t\t%s
            - Host: \t\t%s
            - Bucket name: \t%s
            - Bucket URL: \t%s
        """, domain, str(port), host, name, url)
    return host, url, name

def get_model_data_config():
    """
    Get configuration for model and test data.

    :return:
    """
    path_test_data_dir = os.getenv("PATH_TO_TEST_DATA_DIR", "../data/PS_test_mix/*")
    path_to_model = os.getenv("PATH_TO_AI_MODEL", "../data/Autoencoder_weights_new.h5")
    logger.info(
        """
        Model data configuration:
            - Test data directory: \t%s
            - Model path: \t\t\t%s
        """, path_test_data_dir, path_to_model)
    return path_test_data_dir, path_to_model