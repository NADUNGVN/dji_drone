import socket
import time

TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889
TELLO_ADDR = (TELLO_IP, TELLO_PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 8889))
sock.settimeout(5)

def send(cmd):
    print(">", cmd)
    sock.sendto(cmd.encode("utf-8"), TELLO_ADDR)
    try:
        data, _ = sock.recvfrom(1024)
        print("<", data.decode("utf-8"))
    except socket.timeout:
        print("< timeout: không nhận phản hồi")

send("command")   # vào SDK mode
send("battery?")  # xem pin
send("speed?")    # xem tốc độ

# Khi đã chắc chắn an toàn, có thể thử:
# send("takeoff")
# time.sleep(3)
# send("land")

sock.close()