import unittest
from test_topology import deployment, bindings
from topology import compile_deployment

class ContextSpellingTests(unittest.TestCase):
    def test_both_supported_spellings_preserve_original(self):
        for key in ('max-model-len','max_model_len'):
            dep=deployment();args=dep['instances'][0]['launch']['args']
            del args['max-model-len'];args[key]=4096
            cmd=compile_deployment(dep,bindings())['groups']['a/model']['commands'][0]['argv']
            self.assertEqual(cmd[cmd.index('--'+key)+1],'4096')
            other='max_model_len' if key=='max-model-len' else 'max-model-len'
            self.assertNotIn('--'+other,cmd)
    def test_duplicates_rejected_even_if_equal(self):
        for value in (4096,8192):
            dep=deployment();dep['instances'][0]['launch']['args']['max_model_len']=value
            with self.assertRaisesRegex(ValueError,'Duplicate max-model-len/max_model_len'):
                compile_deployment(dep,bindings())
    def test_invalid_alias_value_rejected(self):
        for value in (0,-1,True,'4096'):
            dep=deployment();args=dep['instances'][0]['launch']['args']
            del args['max-model-len'];args['max_model_len']=value
            with self.assertRaisesRegex(ValueError,'Missing positive max-model-len'):
                compile_deployment(dep,bindings())
if __name__=='__main__':unittest.main()
