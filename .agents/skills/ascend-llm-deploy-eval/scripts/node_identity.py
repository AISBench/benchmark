"""Stable physical node identities beneath a logical job archive."""
import re
from pathlib import PurePosixPath as Path

def node_job_id(job_id,node_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',node_id):
        raise ValueError('Node ID must be 1-32 ASCII letters/digits/underscore/hyphen')
    return job_id+'-node-'+node_id

def node_root(binding,job_id,node_id):
    node_job_id(job_id,node_id)  # Validate before using the directory component.
    return Path(binding['work_dir'])/job_id/'nodes'/node_id
