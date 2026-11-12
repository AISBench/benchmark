# Detached Python runtime (CLI 0.1.0a4)

start-task starts a Linux Python process with a new session, closed inherited descriptors, DEVNULL stdin and append-only logs under TASK/logs/runtime.log. No service manager is required. The runtime owns a task flock and Unix socket. Startup records nonce and identity before launch; a process receipt stores PID, /proc start ticks, boot ID and session. A live socket handshake must match the complete frozen identity and PID. Retrying the same nonce reconciles existing evidence and never blindly restarts an uncertain launch.

SSH clients may exit without stopping the runtime. Processes are not automatically restored after a server reboot. Missing process or socket evidence requires recovery inspection. Cancellation and idle release are runtime functions; stop-runtime is not a substitute for resource release.

After release/cancellation has completed, stop-runtime --request FILE accepts task_ref and the original nonce, with --config. It checks terminal task state and exact process identity, then sends an authenticated socket shutdown request and verifies process exit. Preserve all task files. An unknown stop result is inspected, never handled by guessing PIDs or names.
