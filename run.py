import threading
import webbrowser

from app import app


def open_browser() -> None:
    webbrowser.open("http://localhost:5000", new=1)


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
