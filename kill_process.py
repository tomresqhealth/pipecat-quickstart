# kill_process.py
import subprocess
import os
import signal

def kill_port(port):
    try:
        # Find PIDs using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        
        pids = result.stdout.strip().split('\n')
        
        if not pids or pids == ['']:
            print(f"No process found on port {port}")
            return
        
        for pid_str in pids:
            if pid_str:
                pid = int(pid_str)
                print(f"Killing process {pid} on port {port}...")
                os.kill(pid, signal.SIGKILL)
                print(f"Process {pid} killed.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    kill_port(7860)
    # Also kill any lingering Python processes running bot.py
    subprocess.run(["pkill", "-f", "bot.py"], capture_output=True)
    print("Done.")