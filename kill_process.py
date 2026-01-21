import subprocess
import os
import signal
import sys

def kill_port(port):
    try:
        # Run lsof to find the PID
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"No process found on port {port} or lsof failed.")
            return

        pids = result.stdout.strip().split('\n')
        for pid_str in pids:
            if not pid_str:
                continue
            try:
                pid = int(pid_str)
                print(f"Killing process {pid} on port {port}...")
                os.kill(pid, signal.SIGKILL)
                print(f"Process {pid} killed.")
            except ValueError:
                continue
            except ProcessLookupError:
                print(f"Process {pid} already gone.")
            except PermissionError:
                print(f"Permission denied to kill process {pid}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    kill_port(7860)

