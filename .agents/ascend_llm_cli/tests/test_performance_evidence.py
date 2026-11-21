import json
from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.native_results import validate_native_result
from deploy_eval_v2.performance_evidence import summarize


class PerformanceEvidenceTests(unittest.TestCase):
    def fixture(self,folder,steady=False,timing=True):
        root=Path(folder)
        (root/'execution.json').write_text('{"returncode":0}')
        (root/'native-expectations.json').write_text('[["service","dataset"]]')
        path=root/'outputs/run/performances/service/dataset_details.jsonl';path.parent.mkdir(parents=True)
        rows=[]
        for i in range(20):
            row=dict(id=i,success=True,input_tokens=10,output_tokens=5)
            if timing: row.update(start_time=i*.5,end_time=i*.5+.8)
            rows.append(json.dumps(row))
        path.write_text('\n'.join(rows))
        log=root/'outputs/run/logs/worker.log';log.parent.mkdir(parents=True);log.write_text('evidence')
        stage='stable' if steady else 'total'
        data={key:{stage:value} for key,value in {'Benchmark Duration':'8500 ms' if steady else '10300 ms','Total Requests':18 if steady else 20,
              'Success Requests':18 if steady else 20,'Failed Requests':0,'Max Concurrency':2,
              'Request Throughput':'2.1176 req/s' if steady else '1.9417 req/s'}.items()}
        summary=path.with_name('dataset.json');summary.write_text(json.dumps(data))
        case=dict(formal_requests=20,concurrency=2,steady_state=steady)
        native=validate_native_result(root,case)
        return root,path,summary,data,case,native

    def test_native_summary_and_timing_evidence_can_be_verified(self):
        for steady in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                root,_,_,_,case,native=self.fixture(folder,steady)
                report=summarize(root,case,native)
                self.assertEqual(report['status'],'VERIFIED')
                self.assertEqual(native['request_groups'][0]['observed_tokens']['input_tokens']['total'],200)

    def test_missing_timing_short_window_and_modified_details_are_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            root,path,summary,data,case,native=self.fixture(folder,True,False)
            self.assertEqual(summarize(root,case,native)['groups'][0]['reason'],'FULL_REQUEST_TIMING_UNAVAILABLE')
            path.write_text(path.read_text()+'\n')
            self.assertEqual(summarize(root,case,native)['groups'][0]['reason'],'DETAIL_EVIDENCE_CHANGED')
        with tempfile.TemporaryDirectory() as folder:
            root,_,summary,data,case,native=self.fixture(folder,True)
            data['Benchmark Duration']['stable']='1000 ms';data['Request Throughput']['stable']='18 req/s'
            summary.write_text(json.dumps(data))
            self.assertEqual(summarize(root,case,native)['groups'][0]['reason'],'SUMMARY_TIMING_MISMATCH')

    def test_real_short_stable_interval_is_not_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root,path,summary,data,case,_=self.fixture(folder,True)
            path.write_text('\n'.join(path.read_text().splitlines()[:4]))
            case['formal_requests']=4
            native=validate_native_result(root,case)
            data['Benchmark Duration']['stable']='500 ms'
            data['Request Throughput']['stable']='4 req/s'
            data['Total Requests']['stable']=2;data['Success Requests']['stable']=2
            summary.write_text(json.dumps(data))
            self.assertEqual(summarize(root,case,native)['groups'][0]['reason'],'STEADY_WINDOW_TOO_SHORT')

    def test_summary_count_and_throughput_must_agree(self):
        for field,value in [('Total Requests',19),('Failed Requests',1),('Request Throughput','99 req/s'),
                            ('Benchmark Duration','nan ms'),('Max Concurrency',4)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                root,_,summary,data,case,native=self.fixture(folder)
                data[field]['total']=value;summary.write_text(json.dumps(data))
                self.assertEqual(summarize(root,case,native)['status'],'UNVERIFIED')
