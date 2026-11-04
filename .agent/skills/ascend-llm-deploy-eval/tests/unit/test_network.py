import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import task_preflight as preflight


def nic(name, ip='10.1.0.1', kind=None, state='UP', family='inet'):
    return {'ifname':name,'operstate':state,'linkinfo':{'info_kind':kind},
            'addr_info':[{'family':family,'scope':'global','local':ip}]}


class NetworkTests(unittest.TestCase):
    def select(self, interfaces, speeds, preference=None):
        with patch.object(preflight,'command',return_value=json.dumps(interfaces)), \
             patch.object(preflight,'interface_speed',side_effect=lambda n:speeds.get(n)):
            return preflight.discover_network(preference)

    def test_fastest_usable_ipv4_without_name_assumption(self):
        rows=[nic('management'),nic('fast-net','10.2.0.1','vlan'),
              nic('faster-v6','2001:db8::1',family='inet6'),nic('bridge','10.3.0.1','bridge')]
        result=self.select(rows,{'management':1000,'fast-net':200000,'faster-v6':400000,'bridge':800000})
        self.assertEqual(result['interface'],'fast-net')
        self.assertEqual(result['speed_mbps'],200000)
        self.assertEqual(len(result['candidates']),2)

    def test_ties_and_unknown_speed_require_verification(self):
        for speeds in [{'a':1000,'b':1000},{'a':1000,'b':None}]:
            with self.subTest(speeds=speeds),self.assertRaises(ValueError):
                self.select([nic('a'),nic('b','10.2.0.1')],speeds)

    def test_override_still_requires_active_unique_ipv4(self):
        rows=[nic('a'),nic('b','10.2.0.1',state='DOWN')]
        self.assertEqual(self.select(rows,{'a':None},'a')['interface'],'a')
        with self.assertRaises(ValueError):self.select(rows,{'b':1000},'b')
        rows[0]['addr_info'].append({'family':'inet','scope':'global','local':'10.1.0.2'})
        with self.assertRaises(ValueError):self.select(rows,{'a':1000},'a')

    def test_speed_uses_sysfs_then_current_ethtool_speed(self):
        with patch.object(Path,'read_text',return_value='200000'):
            self.assertEqual(preflight.interface_speed('test0'),200000)
        with patch.object(Path,'read_text',side_effect=OSError()),patch.object(preflight,'command',return_value='Supported: 400000Mb/s\n Speed: 100000Mb/s\n'):
            self.assertEqual(preflight.interface_speed('test0'),100000)
        with patch.object(Path,'read_text',return_value='-1'),patch.object(preflight,'command',side_effect=RuntimeError()):
            self.assertIsNone(preflight.interface_speed('test0'))
