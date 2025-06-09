#!/usr/bin/env python3
"""
Simple HTTP server to serve the Nomad Booster frontend
Runs on port 3000 and serves index.html
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow WebSocket connections to localhost:8000
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        # Handle SPA routing - serve index.html for any path that doesn't exist as a file
        if self.path == '/':
            self.path = '/index.html'
        else:
            # Check if the requested path exists as a file
            file_path = self.path.lstrip('/')
            if not os.path.exists(file_path) and not file_path.startswith('.'):
                # If file doesn't exist and it's not a hidden file, serve index.html for SPA routing
                self.path = '/index.html'
        
        super().do_GET()

def main():
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Create the server
    handler = CustomHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"🌍 Nomad Booster Frontend Server")
            print(f"📡 Serving at http://localhost:{PORT}")
            print(f"📁 Serving files from: {frontend_dir}")
            print(f"🔗 Make sure your API is running on http://localhost:8000")
            print(f"🚀 Opening browser...")
            
            # Open browser automatically
            webbrowser.open(f'http://localhost:{PORT}')
            
            print(f"✅ Server started. Press Ctrl+C to stop.")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Please stop any other servers on this port.")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 