"""Bounded UTF-8 artifact slices; offsets always refer to original file bytes."""
from pathlib import Path
import hashlib
import re
from .protocol import encode, envelope, ProtocolError


def read_result(root, task_id, artifact_ref, offset=0, max_bytes=8192, *, tail_bytes=None,
                line_start=None,line_end=None,keyword=None,scan_offset=0,scan_line=1,expected_sha256=None):
    if type(offset) is not int or offset < 0 or type(max_bytes) is not int or not 1024 <= max_bytes <= 32768:
        raise ProtocolError('INVALID_INPUT')
    modes=sum(value is not None for value in (tail_bytes,line_start,keyword))
    if modes>1 or (modes and offset) or (line_end is not None and line_start is None):
        raise ProtocolError('INVALID_INPUT: mutually exclusive artifact locators')
    if (type(scan_offset) is not int or scan_offset<0 or type(scan_line) is not int or scan_line<1 or
            (tail_bytes is not None and (type(tail_bytes) is not int or not 1<=tail_bytes<=32768)) or
            (line_start is not None and (type(line_start) is not int or line_start<1 or
                type(line_end) is not int or line_end<line_start)) or
            (keyword is not None and (not isinstance(keyword,str) or not 1<=len(keyword.encode('utf-8'))<=256))):
        raise ProtocolError('INVALID_INPUT: artifact locator')
    root = Path(root).resolve()
    relative = Path(artifact_ref)
    if not relative.parts or relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('results', 'logs'):
        raise ProtocolError('INVALID_INPUT: artifact path')
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ProtocolError('INVALID_INPUT: artifact escape')
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        if expected_sha256 is not None:
            if not isinstance(expected_sha256,str) or not re.fullmatch('[a-f0-9]{64}',expected_sha256):
                raise ProtocolError('INVALID_INPUT: evidence digest')
            if relative.parts[0]!='results' or size>4*1024*1024:
                raise ProtocolError('INVALID_INPUT: digest verification requires result artifact up to 4 MiB')
            stream.seek(0)
            hasher=hashlib.sha256()
            hashed=0
            for block in iter(lambda:stream.read(65536),b''):
                hashed+=len(block)
                if hashed>4*1024*1024:
                    raise ProtocolError('RECOVERY_REQUIRED: result changed during verification')
                hasher.update(block)
            if hasher.hexdigest()!=expected_sha256:
                raise ProtocolError('RECOVERY_REQUIRED: result evidence digest mismatch')
        locator={}
        range_end=size
        selected_line=scan_line
        if tail_bytes is not None:
            offset=max(0,size-tail_bytes)
            stream.seek(offset)
            while offset<size:
                byte=stream.read(1)
                if byte[0]&0xC0!=0x80:
                    break
                offset+=1
            locator=dict(mode='tail',requested_tail_bytes=tail_bytes)
        elif keyword is not None or line_start is not None:
            if scan_offset>size:
                raise ProtocolError('INVALID_INPUT: scan offset beyond EOF')
            stream.seek(scan_offset)
            block=stream.read(1024*1024)
            if keyword is not None:
                needle=keyword.encode('utf-8')
                found=block.find(needle)
                offset=scan_offset+found if found>=0 else scan_offset
                resume=scan_offset+len(block) if scan_offset+len(block)>=size else scan_offset+max(0,len(block)-len(needle)+1)
                locator=dict(mode='keyword',matched=found>=0,next_scan_offset=resume,
                             scan_complete=scan_offset+len(block)>=size)
            else:
                index=0
                while selected_line<line_start and index<len(block):
                    newline=block.find(b'\n',index)
                    if newline<0:
                        index=len(block)
                        break
                    selected_line+=1
                    index=newline+1
                matched=selected_line>=line_start and selected_line<=line_end and scan_offset+index<size
                offset=scan_offset+index
                locator=dict(mode='lines',matched=matched,line_start=line_start,line_end=line_end,
                             next_scan_offset=offset,next_scan_line=selected_line,
                             scan_complete=scan_offset+len(block)>=size)
                if matched:
                    end=index
                    count=selected_line
                    while count<=line_end:
                        newline=block.find(b'\n',end)
                        if newline<0:
                            break
                        end=newline+1;count+=1
                    if count>line_end:
                        range_end=scan_offset+end
            if not locator['matched']:
                result=envelope(task_id,0)
                if expected_sha256 is not None:
                    result['verified_sha256']=expected_sha256
                result.update(artifact_ref=artifact_ref,total_bytes=size,content='',actual_start=offset,
                              actual_end=offset,next_offset=offset,truncated=not locator['scan_complete'],locator=locator)
                if len(encode(result))>max_bytes:
                    raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: metadata exceeds budget')
                return result
        if offset > size:
            raise ProtocolError('INVALID_INPUT: offset beyond EOF')
        stream.seek(offset)
        data = stream.read(min(max_bytes,range_end-offset))
    if data and data[0] & 0xC0 == 0x80:
        raise ProtocolError('INVALID_INPUT: offset splits UTF-8 character')
    # At most a three-byte incomplete suffix can be caused by the bounded read.
    for trim in range(4):
        try:
            text = (data[:-trim] if trim else data).decode('utf-8')
            break
        except UnicodeDecodeError:
            if trim == 3:
                raise ProtocolError('INVALID_INPUT: artifact is not UTF-8')
    result = envelope(task_id, 0)
    if expected_sha256 is not None:
        result['verified_sha256']=expected_sha256
    result.update(artifact_ref=artifact_ref, offset=offset, actual_start=offset,
                  total_bytes=size, truncated=True, next_offset=offset, content='')
    if locator:
        result['locator']=locator
    low, high = 0, len(text)
    while low < high:
        middle = (low+high+1)//2
        candidate = text[:middle]
        end = offset+len(candidate.encode('utf-8'))
        result.update(content=candidate, next_offset=end, actual_end=end, truncated=end < range_end)
        if line_start is not None:
            locator.update(next_scan_offset=end,next_scan_line=selected_line+candidate.count('\n'),scan_complete=end>=range_end)
        elif keyword is not None:
            locator.update(next_scan_offset=end if end>=size else max(offset+1,end-len(keyword.encode('utf-8'))+1),scan_complete=end>=size)
        if len(encode(result)) <= max_bytes:
            low = middle
        else:
            high = middle-1
    end = offset+len(text[:low].encode('utf-8'))
    result.update(content=text[:low], next_offset=end, actual_end=end, truncated=end < range_end)
    if line_start is not None:
        locator.update(next_scan_offset=end,next_scan_line=selected_line+text[:low].count('\n'),scan_complete=end>=range_end)
    elif keyword is not None:
        locator.update(next_scan_offset=end if end>=size else max(offset+1,end-len(keyword.encode('utf-8'))+1),scan_complete=end>=size)
    if len(encode(result)) > max_bytes or (data and end == offset):
        raise ProtocolError('TRANSPORT_PROTOCOL_ERROR: metadata exceeds budget')
    return result
