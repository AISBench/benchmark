# Disposable client error semantics

The SSH-side client validates command/query fields before connecting, sends one
bounded JSON request and reads one response bounded by the requested envelope
limit. Ordinary responses are at most 8 KiB; explicit artifact reads may use
32 KiB. Task/protocol identity, status, JSON validity and newline framing are
checked before forwarding. Duplicate keys and non-finite values are rejected by
both client and runtime request parsing.

Socket loss produces UNKNOWN_OUTCOME, not operation.failed. The JSON error retains
the supplied after_seq and valid request/operation identity. It neither cancels
the stage nor resubmits it. Argument and framing failures also produce bounded
structured output rather than a full traceback on SSH stderr. The host's lack of
Unix sockets is reported explicitly as SSH_UNAVAILABLE.

Local tests inject socket IO to cover framing, wrong identity, oversize responses,
connection loss and preserved cursors. The Windows Python in this workspace lacks
AF_UNIX and no WSL installation is available. Linux client/runtime integration and
independent systemd survival remain unverified and require a suitable environment.
