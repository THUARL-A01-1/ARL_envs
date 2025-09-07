import numpy as np
import socket

HOST = 'localhost'
ANY6D_PORT = 5000
CLIP_PORT = 6000
ANY6D_REQUIRED = ["color", "depth", "object_name", "task"]
CLIP_REQUIRED = ["color"]

class Client:
    """
    A generic UDP client for sending requests and receiving responses from a server.
    Subclasses should implement format_request and parse_response.
    """
    def __init__(self, server='any6d', timeout=10.0):
        if server not in ['any6d', 'clip']:
            raise ValueError(f"PoseClient: unknown server '{server}'.")
        
        if server == 'any6d':
            self.required = ANY6D_REQUIRED
            self.server_address = (HOST, ANY6D_PORT)
        elif server == 'clip':
            self.required = CLIP_REQUIRED
            self.server_address = (HOST, CLIP_PORT)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

    def __call__(self, **kwds):
        message = self.format_request(**kwds)
        try:
            self.sock.sendto(message, self.server_address)
            data, _ = self.sock.recvfrom(4096)
            return self.parse_response(data)
        except socket.timeout:
            print(f"{self.__class__.__name__}: socket timeout.")
            return None

    def format_request(self, **kwds):
        for key in self.required:
            if key not in kwds:
                raise ValueError(f"PoseClient: missing required argument '{key}'.")
        msg = f"{','.join([str(kwds[key]) for key in self.required])}"
        
        return msg.encode()

    def parse_response(self, data):
        response = data.decode()
        arr = np.fromstring(response, sep=',')

        return arr