import unittest
from unittest.mock import Mock,patch
from deploy_eval_v2.durable import rename_new_directory
from deploy_eval_v2.protocol import ProtocolError


class DirectoryPublicationTests(unittest.TestCase):
    def test_windows_access_denial_retries_only_rename(self):
        error=PermissionError('sharing denied');error.winerror=5
        source=Mock();source.rename.side_effect=[error,None]
        destination=Mock();destination.exists.return_value=False
        with patch('deploy_eval_v2.durable.os.name','nt'):
            rename_new_directory(source,destination)
        self.assertEqual(source.rename.call_count,2)

    def test_destination_appearing_during_retry_is_not_overwritten(self):
        error=PermissionError('sharing denied');error.winerror=32
        source=Mock();source.rename.side_effect=error
        destination=Mock();destination.exists.side_effect=[False,True]
        with patch('deploy_eval_v2.durable.os.name','nt'):
            with self.assertRaisesRegex(ProtocolError,'REQUEST_CONFLICT'):
                rename_new_directory(source,destination)
        source.rename.assert_called_once()
