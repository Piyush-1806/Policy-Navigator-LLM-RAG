import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 8000)}"
backlog = 2048

# Worker processes
workers = min(4, (multiprocessing.cpu_count() * 2) + 1)  # Limit workers for memory
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'hackrx-llm-system'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (disabled for Render)
keyfile = None
certfile = None

# Application
module = "main:app"

# Memory and performance
preload_app = True  # Preload for better memory efficiency
enable_stdio_inheritance = True

def when_ready(server):
    server.log.info("🚀 HackRx LLM System is ready!")
    server.log.info(f"🌐 Listening on {bind}")
    server.log.info(f"👷 Workers: {workers}")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")
    
def pre_fork(server, worker):
    server.log.info(f"Worker {worker.pid} spawned")

def post_fork(server, worker):
    server.log.info(f"Worker {worker.pid} spawned")

def worker_exit(server, worker):
    server.log.info(f"Worker {worker.pid} exited")