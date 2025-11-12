"""Shared constants for communication between tasks and inferencers."""
import struct
from collections import OrderedDict


STATUS_REPORT_INTERVAL = 1
WAIT_FLAG = 2
SYNC_MAIN_PROCESS_INTERVAL = 0.1


class _MessageInfo:
    STATUS = None
    POST = None
    RECV = None
    FAIL = None
    FINISH = None
    DATA_SYNC_FLAG = None
    DATA_INDEX = None


MESSAGE_INFO = _MessageInfo()

FIELDS = OrderedDict(
    [
        ("STATUS", "I"),
        ("POST", "I"),
        ("RECV", "I"),
        ("FAIL", "I"),
        ("FINISH", "I"),
        ("DATA_SYNC_FLAG", "I"),
        ("DATA_INDEX", "i"),
    ]
)

# Calculate offsets for each field
offset = 0
for name, fmt in FIELDS.items():
    size = struct.calcsize(fmt)
    setattr(MESSAGE_INFO, name, (offset, offset + size))
    offset += size

