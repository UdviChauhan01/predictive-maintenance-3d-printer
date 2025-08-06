import serial
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import threading
import queue

class PrinterDataCollector:
    """
    Data collector for 3D printer sensors and operational data
    """
    
    def __init__(self, port: str = 'COM3', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.data_queue = queue.Queue()
        self.is_collecting = False
        self.collection_thread = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('printer_data.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Sensor data structure
        self.sensor_data = {
            'timestamp': None,
            'temperature': {
                'hotend': None,
                'bed': None,
                'ambient': None
            },
            'vibration': {
                'x_axis': None,
                'y_axis': None,
                'z_axis': None
            },
            'motor_current': {
                'x_motor': None,
                'y_motor': None,
                'z_motor': None,
                'extruder': None
            },
            'position': {
                'x': None,
                'y': None,
                'z': None
            },
            'print_status': {
                'is_printing': False,
                'layer_height': None,
                'print_progress': None,
                'filament_used': None
            },
            'maintenance_indicators': {
                'belt_tension': None,
                'nozzle_wear': None,
                'bed_level': None,
                'extruder_clogging': None
            }
        }
    
    def connect_serial(self) -> bool:
        """Establish serial connection to 3D printer"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.logger.info(f"Connected to printer on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to printer: {e}")
            return False
    
    def send_gcode(self, command: str) -> str:
        """Send G-code command to printer and get response"""
        if not self.serial_conn:
            return ""
        
        try:
            self.serial_conn.write(f"{command}\n".encode())
            time.sleep(0.1)
            response = self.serial_conn.readline().decode().strip()
            return response
        except Exception as e:
            self.logger.error(f"Error sending G-code {command}: {e}")
            return ""
    
    def get_temperature_data(self) -> Dict:
        """Get current temperature readings"""
        try:
            # Send M105 command to get temperature
            response = self.send_gcode("M105")
            
            # Parse temperature response (example: "ok T:200.0 /200.0 B:60.0 /60.0")
            if "T:" in response:
                temp_parts = response.split()
                hotend_temp = None
                bed_temp = None
                
                for part in temp_parts:
                    if part.startswith("T:"):
                        hotend_temp = float(part.split(":")[1].split("/")[0])
                    elif part.startswith("B:"):
                        bed_temp = float(part.split(":")[1].split("/")[0])
                
                return {
                    'hotend': hotend_temp,
                    'bed': bed_temp,
                    'ambient': 25.0  # Default ambient temperature
                }
        except Exception as e:
            self.logger.error(f"Error getting temperature data: {e}")
        
        return {'hotend': None, 'bed': None, 'ambient': None}
    
    def get_position_data(self) -> Dict:
        """Get current position data"""
        try:
            response = self.send_gcode("M114")
            
            # Parse position response (example: "X:100.00 Y:100.00 Z:0.20")
            if "X:" in response:
                pos_parts = response.split()
                x_pos = y_pos = z_pos = None
                
                for part in pos_parts:
                    if part.startswith("X:"):
                        x_pos = float(part.split(":")[1])
                    elif part.startswith("Y:"):
                        y_pos = float(part.split(":")[1])
                    elif part.startswith("Z:"):
                        z_pos = float(part.split(":")[1])
                
                return {'x': x_pos, 'y': y_pos, 'z': z_pos}
        except Exception as e:
            self.logger.error(f"Error getting position data: {e}")
        
        return {'x': None, 'y': None, 'z': None}
    
    def simulate_sensor_data(self) -> Dict:
        """Simulate sensor data for testing purposes"""
        import random
        
        # Simulate realistic 3D printer data
        base_temp = 200 + random.uniform(-5, 5)
        base_bed = 60 + random.uniform(-2, 2)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'temperature': {
                'hotend': base_temp,
                'bed': base_bed,
                'ambient': 25 + random.uniform(-3, 3)
            },
            'vibration': {
                'x_axis': random.uniform(0.1, 0.5),
                'y_axis': random.uniform(0.1, 0.5),
                'z_axis': random.uniform(0.05, 0.2)
            },
            'motor_current': {
                'x_motor': random.uniform(0.8, 1.2),
                'y_motor': random.uniform(0.8, 1.2),
                'z_motor': random.uniform(0.3, 0.7),
                'extruder': random.uniform(0.5, 1.0)
            },
            'position': {
                'x': random.uniform(0, 200),
                'y': random.uniform(0, 200),
                'z': random.uniform(0, 200)
            },
            'print_status': {
                'is_printing': random.choice([True, False]),
                'layer_height': random.uniform(0.1, 0.3),
                'print_progress': random.uniform(0, 100),
                'filament_used': random.uniform(0, 100)
            },
            'maintenance_indicators': {
                'belt_tension': random.uniform(0.7, 1.0),
                'nozzle_wear': random.uniform(0, 0.3),
                'bed_level': random.uniform(-0.1, 0.1),
                'extruder_clogging': random.uniform(0, 0.2)
            }
        }
    
    def collect_data_continuously(self):
        """Continuously collect data in a separate thread"""
        while self.is_collecting:
            try:
                # Try to get real data first
                if self.serial_conn and self.serial_conn.is_open:
                    temp_data = self.get_temperature_data()
                    pos_data = self.get_position_data()
                    
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'temperature': temp_data,
                        'position': pos_data,
                        # Add simulated data for other sensors
                        'vibration': self.sensor_data['vibration'],
                        'motor_current': self.sensor_data['motor_current'],
                        'print_status': self.sensor_data['print_status'],
                        'maintenance_indicators': self.sensor_data['maintenance_indicators']
                    }
                else:
                    # Use simulated data if no connection
                    data = self.simulate_sensor_data()
                
                self.data_queue.put(data)
                time.sleep(1)  # Collect data every second
                
            except Exception as e:
                self.logger.error(f"Error in data collection: {e}")
                time.sleep(5)  # Wait before retrying
    
    def start_collection(self):
        """Start continuous data collection"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self.collect_data_continuously)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            self.logger.info("Started data collection")
    
    def stop_collection(self):
        """Stop continuous data collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        self.logger.info("Stopped data collection")
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get the latest collected data"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def save_data_to_file(self, filename: Optional[str] = None):
        """Save collected data to JSON file"""
        if filename is None:
            filename = f"printer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data_list = []
        while not self.data_queue.empty():
            data = self.data_queue.get()
            data_list.append(data)
        
        if data_list:
            with open(filename, 'w') as f:
                json.dump(data_list, f, indent=2)
            self.logger.info(f"Saved {len(data_list)} data points to {filename}")
    
    def close(self):
        """Close the data collector"""
        self.stop_collection()
        if self.serial_conn:
            self.serial_conn.close()
        self.logger.info("Data collector closed")

if __name__ == "__main__":
    # Example usage
    collector = PrinterDataCollector()
    
    try:
        # Try to connect to printer
        if not collector.connect_serial():
            print("Using simulated data (no printer connection)")
        
        # Start data collection
        collector.start_collection()
        
        # Collect data for 60 seconds
        print("Collecting data for 60 seconds...")
        time.sleep(60)
        
        # Save collected data
        collector.save_data_to_file()
        
    finally:
        collector.close() 
