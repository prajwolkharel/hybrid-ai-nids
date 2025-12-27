from nfstream import NFPlugin
import psutil

print("Available network interfaces:")
for iface in psutil.net_if_addrs().keys():
    print(f"  - {iface}")