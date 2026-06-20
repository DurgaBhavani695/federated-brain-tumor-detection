import uvicorn
import webbrowser
from threading import Timer
import sys

def open_browser():
    print("Launching browser dashboard...")
    webbrowser.open_new("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("==================================================================")
    print("   PRIVACY-PRESERVING FEDERATED BRAIN TUMOR DETECTION PLATFORM   ")
    print("==================================================================")
    print("Starting FastAPI Uvicorn Server on http://127.0.0.1:8000")
    
    # Launch browser after 1.5 seconds to let uvicorn initialize first
    Timer(1.5, open_browser).start()
    
    try:
        # Run server. Disable hot-reload to prevent multiple browser window spawns
        uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info", reload=False)
    except KeyboardInterrupt:
        print("\nShutting down platform. Goodbye!")
        sys.exit(0)
