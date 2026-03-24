# communication.py (worker) - Versión corregida si es necesario
import asyncio
import json
import struct
from typing import Optional, Dict, Any

async def send_json(writer: asyncio.StreamWriter, data: dict):
    """Envía datos JSON con prefijo de longitud."""
    message = json.dumps(data).encode()
    length = struct.pack(">I", len(message))
    writer.write(length + message)
    await writer.drain()

async def recv_json(reader: asyncio.StreamReader) -> Optional[dict]:
    """Recibe datos JSON con prefijo de longitud."""
    try:
        raw_length = await reader.read(4)
        if not raw_length or len(raw_length) < 4:
            return None
        
        message_length = struct.unpack(">I", raw_length)[0]
        
        # Evitar mensajes demasiado grandes
        if message_length > 100 * 1024 * 1024:  # 100 MB máximo
            print(f" Mensaje demasiado grande: {message_length} bytes")
            return None
        
        data = b""
        while len(data) < message_length:
            chunk_size = min(4096, message_length - len(data))
            packet = await reader.read(chunk_size)
            if not packet:
                return None
            data += packet
        
        return json.loads(data.decode())
    
    except Exception as e:
        print(f"[ERROR Communication] {e}")
        return None

async def connect_to_server(ip: str, port: int):
    """Establece conexión con el servidor."""
    print(f"\nConectando al servidor {ip}:{port}...")
    try:
        reader, writer = await asyncio.open_connection(ip, port)
        print(" Conectado al servidor")
        return reader, writer
    except Exception as e:
        print(f" Error de conexión: {e}")
        return None, None