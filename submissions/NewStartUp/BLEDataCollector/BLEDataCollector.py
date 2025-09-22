import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice

import BLEDataCollector


def run_ble_collector(collector: BLEDataCollector, device_name: str):
    """Target function for the thread. Runs the asyncio event loop."""
    try:
        asyncio.run(collector.scan_and_connect(auto_connect_device=device_name))
    except Exception as e:
        print(f"Error in BLE thread: {e}")

class BLEDataCollector:

    def __init__(self):
        self.count: int = 0
        self.rate: float = 0.0
        self.start_time: float = 0.0
        self.processed_data: Dict[str, List[Any]] = {'A': [], 'B': []}
        self.is_running = True  # Flag to control the connection loop

        report_dir = os.path.join(os.getcwd(), "report")
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        log_path = os.path.join(report_dir, f"{uuid.uuid4()}.txt")
        self.log_file = open(log_path, "w")
        print(f"Logging data to {log_path}")

    def _get_datetime(self) -> str:
        return datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")

    def log(self, message: str, title: str = "INFO"):
        log_entry = f"{self._get_datetime()} : {title} \t {message}\n"
        self.log_file.write(log_entry)

    def _on_data_received(self, sender: BleakGATTCharacteristic, data: bytearray):
        self.count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.rate = self.count / elapsed_time
            self.count = 0
            self.start_time = time.time()
        try:
            res = json.loads(data.decode())
            self.processed_data['A'].extend(res.get('A', []))
            self.processed_data['B'].extend(res.get('B', []))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.log(title="ERROR", message=f"Failed to decode data: {e}")

    # MODIFIED for Pygame integration
    async def connect_device(self, device: BLEDevice):
        print(f"Connecting to {device.name} ({device.address})")
        async with BleakClient(device.address) as client:
            print(f"Connected: {client.is_connected}")
            notify_char = next((c for s in client.services for c in s.characteristics if "notify" in c.properties),
                               None)
            write_char = next((c for s in client.services for c in s.characteristics if "write" in c.properties), None)

            if notify_char and write_char:
                await client.write_gatt_char(write_char.uuid, bytearray([0x39]), response=True)
                await client.start_notify(notify_char.uuid, self._on_data_received)

                # Loop indefinitely, allowing the connection to persist while Pygame runs
                while self.is_running:
                    await asyncio.sleep(0.1)  # Yield control, checking the flag every 100ms

                await client.stop_notify(notify_char.uuid)
            else:
                print("Error: Could not find required characteristics.")
        print(f"{device.name} disconnected.")

    # (scan_and_connect, get_current_data, and close methods are IDENTICAL to the previous version)
    async def scan_and_connect(self, auto_connect_device: str = ""):
        print("Scanning for devices...")
        device = await BleakScanner.find_device_by_name(auto_connect_device)
        if device:
            await self.connect_device(device)
        else:
            print(f"Device '{auto_connect_device}' not found.")

    def get_current_data(self, num_samples: int = 500) -> List[Any]:
        return [
            self.processed_data['A'][-num_samples:],
            self.processed_data['B'][-num_samples:],
        ]

    def close(self):
        if self.log_file and not self.log_file.closed:
            print(f"\nClosing log file: {self.log_file.name}")
            self.log_file.close()

    # NEW method for graceful shutdown
    def stop(self):
        """Signals the async loop to stop."""
        print("Stopping BLE connection...")
        self.is_running = False

    def clear_data(self):
        self.processed_data = {'A': [], 'B': []}
