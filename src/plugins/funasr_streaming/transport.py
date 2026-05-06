from typing import Any

import websockets

connect = websockets.connect
exceptions = websockets.exceptions
ConnectionClosed = websockets.exceptions.ConnectionClosed
WebSocketClientProtocol = Any
WebSocketException = websockets.exceptions.WebSocketException
