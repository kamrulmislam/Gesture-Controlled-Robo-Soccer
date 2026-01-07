import json
import time
from vex import *
from vex.vex_globals import *

# # Robot initialization for AIM platform
# robot = Robot()

def handle_command(robot, control_command, path=None):
    try:
        for message in control_command:
            command = json.loads(message)
            action = command.get("action", "")
            
            if action == "led_on":
                color_name = command.get("color", "BLUE")
                # Map string color names to vex.Color constants
                color_map = {
                    "RED": RED,
                    "GREEN": GREEN,
                    "BLUE": BLUE,
                    "WHITE": WHITE,
                    "YELLOW": YELLOW,
                    "ORANGE": ORANGE,
                    "PURPLE": PURPLE,
                    "CYAN": CYAN,
                }
                color = color_map.get(color_name.upper(), BLUE)  # Default to BLUE if not found
                print(f"Turning LED on with color: {color_name}")
                robot.led.on(ALL_LEDS, color)
                # Send a response back to the client
                return {"status": "success", "action": "led_on", "color": color_name}
                
            elif action == "move":
                distance_inches = command.get("distance", 2)  # Default to 2 inches instead of 100mm
                heading = command.get("heading", 0)
                # Convert inches to millimeters (1 inch = 25.4 mm)
                distance_mm = distance_inches * 25.4
                print(f"Received move command: {command}")
                print(f"Moving robot: Distance={distance_inches} inches ({distance_mm} mm), Heading={heading}")
                robot.move_for(distance_mm, heading)
                # Send a response back to the client
                return {"status": "success", "action": "move", "distance_inches": distance_inches, "distance_mm": distance_mm, "heading": heading}
                
            elif action == "turn_left":
                degrees = command.get("degrees", 90)
                print(f"Turning robot left: {degrees} degrees")
                robot.turn_for(vex.TurnType.LEFT, degrees)  # Correct VEX method
                # Send a response back to the client
                return {"status": "success", "action": "turn_left", "degrees": degrees}
                
            elif action == "turn_right":
                degrees = command.get("degrees", 90)
                print(f"Turning robot right: {degrees} degrees")
                robot.turn_for(vex.TurnType.RIGHT, degrees)  # Correct VEX method
                # Send a response back to the client
                return {"status": "success", "action": "turn_right", "degrees": degrees}
                
            else:
                print(f"Unknown command: {command}")
                # Send an error response back to the client
                return {"status": "error", "message": "Unknown command"}
    except Exception as e:
        print(f"Error handling command: {e}")
        return {"status": "error", "message": str(e)}
    

def send_command_to_vex(robot, command, distance_unit_in_mm=50):
    try:
        robot.led.off(ALL_LEDS)
        if command == "up": # Forward
            robot.move_for(distance_unit_in_mm, 0) # Heading 0 degree
            
        elif command == "down": # Backward
            robot.move_for(distance_unit_in_mm, 180) # Heading 180 degree

        elif command == "left": # Turn Left ===> Move one unit forward
            robot.led.on(1, YELLOW)
            robot.led.on(2, YELLOW)
            robot.turn_for(vex.TurnType.LEFT, 90) # 90 degree turn
            robot.move_for(distance_unit_in_mm, 0) # Heading 0 degree

        elif command == "right": # Turn Right ===> Move one unit forward
            robot.led.on(3, YELLOW)
            robot.led.on(4, YELLOW)
            robot.turn_for(vex.TurnType.RIGHT, 90) # 90 degree turn
            robot.move_for(distance_unit_in_mm, 0) # Heading 0 degree

        elif command == "push": # Kick the ball
            robot.led.on(ALL_LEDS, YELLOW)
            robot.kicker.kick(MEDIUM)

        elif command == "led": # For test purpose
            robot.led.on(3, YELLOW)
            time.sleep(5)

        else:
            return {"status": "error", "message": "unknown_command"}

        return {"status": "success", "message": {command}}
    except Exception as e:
        print(f"Error handling command: {e}")
        return {"status": "error", "message": str(e)}
