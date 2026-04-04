# import modules
import os
import socket
import argparse
import http.server
import socketserver

parser = argparse.ArgumentParser(description='Serve files from the current directory.')

host_name = socket.gethostname()
ip = socket.gethostbyname(host_name)

parser.add_argument('--host', default=ip, type=str, required=False, help='Specify the ip address to listen on.')

parser.add_argument('--port', default=8080, type=int, required=False, help='Specify the port to listen on.')

args = parser.parse_args()

handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer((args.host, args.port), handler) as httpd:
    print(f'Server is listening on {args.host} on port {args.port}.')
    httpd.serve_forever()

# Usases:
# My laptop's IP address is xxx.xxx.xx.xx on port 8080
# so on my phone I navigate to http://xxx.xxx.xx.xx:8080 to access lapto's files
