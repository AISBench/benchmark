import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest
from shared_storage import declared_groups, verify_groups

class DeclarationTests(unittest.TestCase):
    def bindings(self):
        return {'nodes':{'p':{'host':'a','work_dir':'/shared'},'d':{'host':'b','work_dir':'/alias'}},'shared_storage_groups':[{'id':'g','nodes':['p','d']}]}
    def test_default_independent(self):
        b=self.bindings(); del b['shared_storage_groups']; self.assertEqual(declared_groups(b),[])
    def test_alias_declared(self):
        b=self.bindings(); b['nodes']['d']['work_dir']='/alias'
        self.assertEqual(declared_groups(b)[0][1][1], ('b','/alias'))
    def run_probe(self,shared):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); (base/'a').mkdir(); (base/'b').mkdir()
            def ssh(host,argv):
                args=list(argv); self.assertEqual(str(PurePosixPath(args[3]).parent), '/shared' if host=='a' else '/alias'); args[3]=str(base/('a' if shared else host)/Path(args[3]).name)
                return subprocess.check_output([sys.executable,*args[1:]],text=True,stderr=subprocess.PIPE)
            report=base/'report.json'
            if shared: self.assertEqual(len(verify_groups(self.bindings(),ssh,report)),1)
            else:
                with self.assertRaises(subprocess.CalledProcessError): verify_groups(self.bindings(),ssh,report)
            self.assertEqual(json.loads(report.read_text())['status'],'PASSED' if shared else 'FAILED')
            self.assertEqual(list((base/'a').iterdir()),[]); self.assertEqual(list((base/'b').iterdir()),[])
    def test_bidirectional_visible_and_cleaned(self): self.run_probe(True)
    def test_false_declaration_blocks_and_cleans(self): self.run_probe(False)

if __name__=='__main__': unittest.main()
