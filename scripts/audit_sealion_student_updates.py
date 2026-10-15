"""Read-only S3 streaming weight audit; no local model files or paid resources."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from urllib.parse import urlparse

import boto3
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pycon_model2vec_demo.sealion_weight_audit import compare_block


class MatrixReader:
    def __init__(self,response,expected):
        self.body=response["Body"]
        self.digest=sha256()
        self.expected=expected
        if response["ContentLength"] != expected["bytes"]:
            raise ValueError("Weight length mismatch")
        length=struct.unpack("<Q",self.read(8))[0]
        if length>65536:
            raise ValueError("Unexpected safetensors header size")
        header=json.loads(self.read(length))
        tensors=[v for k,v in header.items() if k!="__metadata__"]
        if (len(tensors)!=1 or tensors[0]["dtype"]!="F32"
                or tensors[0]["shape"]!=[249999,1024]
                or tensors[0]["data_offsets"]!=[0,249999*1024*4]):
            raise ValueError("Unexpected static weight structure")

    def read(self,n):
        chunks=[]
        remaining=n
        while remaining:
            block=self.body.read(remaining)
            if not block:
                raise ValueError("Premature weight EOF")
            chunks.append(block)
            self.digest.update(block)
            remaining-=len(block)
        return b"".join(chunks)

    def block(self,rows):
        return np.frombuffer(self.read(rows*1024*4),dtype="<f4").reshape(rows,1024)

    def finish(self):
        if self.body.read(1):
            raise ValueError("Trailing weight bytes")
        if self.digest.hexdigest()!=self.expected["sha256"]:
            raise ValueError("Independent full-byte weight checksum mismatch")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_directory",type=Path)
    args=parser.parse_args()
    record=json.loads((args.run_directory/"resource-record.json").read_text())
    session=boto3.Session(profile_name=record["profile"],region_name=record["region"])
    sm,s3=session.client("sagemaker"),session.client("s3")
    if sm.describe_training_job(TrainingJobName=record["job_name"])["TrainingJobStatus"]!="Completed":
        raise ValueError("Training is not complete")
    uri=urlparse(record["s3_results"])
    prefix=uri.path.strip("/")
    complete=json.loads(s3.get_object(Bucket=uri.netloc,Key=prefix+"/reports/completion.json")["Body"].read())
    artifacts=complete["artifacts"]
    initial_key="models/initialization/model.safetensors"
    initial_spec=artifacts[initial_key]
    if initial_spec["sha256"]!="a58a9d3b0cd6d9682348520797fe9da23bd450e02d1bd28a214e8eb8a6f9bcef":
        raise ValueError("Unexpected shared initialization")
    result={"job_name":record["job_name"],"audit":"all weight rows; official fixed export scaling excluded",
            "not_a_retrieval_benchmark":True,"arms":{}}
    for report in complete["training_reports"]:
        arm=report["arm"]
        tokenizer=artifacts[f"models/{arm}/tokenizer.json"]
        if tokenizer["sha256"]!=artifacts["models/initialization/tokenizer.json"]["sha256"]:
            raise ValueError("Tokenizer identity differs; row comparison unsafe")
        final_key=f"models/{arm}/model.safetensors"
        a=MatrixReader(s3.get_object(Bucket=uri.netloc,Key=prefix+"/"+initial_key),initial_spec)
        b=MatrixReader(s3.get_object(Bucket=uri.netloc,Key=prefix+"/"+final_key),artifacts[final_key])
        totals={}
        try:
            for start in range(0,249999,4096):
                n=min(4096,249999-start)
                summary=compare_block(a.block(n),b.block(n),start,report["pad_id"])
                for name,value in summary.items():
                    totals[name]=totals.get(name,0)+value
            a.finish()
            b.finish()
        finally:
            a.body.close()
            b.body.close()
        if totals["changed_beyond_fixed_export_scaling"]==0:
            raise ValueError("No learned weight change beyond fixed export transformation")
        result["arms"][arm]={**totals,"full_byte_hashes_verified":True,"tokenizer_unchanged":True}
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
