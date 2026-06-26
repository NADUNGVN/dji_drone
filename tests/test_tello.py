from djitellopy import Tello

tello = Tello()
connected = False

try:
    tello.connect()
    connected = True

    print("Battery:", tello.get_battery(), "%")

except Exception as e:
    print("Lỗi connect:", e)

finally:
    if connected:
        tello.end()