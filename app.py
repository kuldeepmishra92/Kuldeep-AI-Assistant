import os
from api.index import app
from keep_alive import start_keep_alive

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    start_keep_alive()   # 🔁 Keeps HuggingFace Space awake (pings /ping every 5 min)
    app.run(host="0.0.0.0", port=port, debug=False)