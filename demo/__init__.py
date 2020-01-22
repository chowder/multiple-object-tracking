import time
import threading
from queue import Queue
from flask import Flask, request

from demo.tracker import RealtimeTrackerDemo
from mot import Observation

app = Flask(__name__)
app.use_reloader=False
task_queue = Queue()


@app.route("/")
def home():
    return "This is the root directory of the Flask application"


@app.route("/data")
def data():
    t = float(request.args.get('time'))
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    task_queue.put(Observation(time=t, x=x, y=y, mac="null", index=0))
    return "OK"


if __name__ == "__main__":
    tracker = RealtimeTrackerDemo()
    threading.Thread(target=app.run).start()
    while True:
        if task_queue.qsize() > 0:
            tracker.add(task_queue.get())
        else:
            time.sleep(1)
