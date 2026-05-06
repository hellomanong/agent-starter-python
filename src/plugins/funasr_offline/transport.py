from typing import Any

import websockets

connect = websockets.connect
ConnectionClosed = websockets.exceptions.ConnectionClosed
WebSocketClientProtocol = Any
WebSocketException = websockets.exceptions.WebSocketException
