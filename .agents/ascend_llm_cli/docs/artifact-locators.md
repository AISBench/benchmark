# Bounded artifact locating

read-result retains byte-offset reads and adds mutually exclusive modes:

    read-result --task-ref t --artifact-ref logs/model.log --tail-bytes 8192
    read-result --task-ref t --artifact-ref logs/model.log --line-start 100 --line-end 120
    read-result --task-ref t --artifact-ref logs/model.log --keyword ERROR

The final serialized response remains at most max_bytes (default 8192, maximum
32768), including metadata and JSON escaping. Tail size is a raw-file selection,
not a promise that all selected text fits one response. Continue raw content with
next_offset. UTF-8 starts/ends never split a returned character.

Line numbers are 1-based. Locating scans at most 1 MiB per call with bounded
memory. When truncated, continue the same line selection with
--scan-offset locator.next_scan_offset and --scan-line locator.next_scan_line.
This handles very long lines without reading the entire line into memory.

Keyword is literal UTF-8, at most 256 encoded bytes. A missing match in one scan
does not imply EOF: check locator.scan_complete. Continue with --scan-offset
locator.next_scan_offset and the same keyword. Scan overlap preserves matches
across chunk boundaries; a delivered match advances only through returned content,
not over later undelivered matches. No regex or recursive filesystem search occurs.

Responses expose actual_start/actual_end, total_bytes, next_offset, truncated and
locator state. Paths remain confined to task logs/results. These modes locate
artifacts only and do not replace the event cursor or change task execution state.
