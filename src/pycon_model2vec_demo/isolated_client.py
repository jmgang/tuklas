"""Checked remote encoding; this client never loads a model or a corpus matrix."""
from __future__ import annotations
import time

from .isolated_serving import check_isolation, decode_response


class RemoteEncoder:
    def __init__(self, send, *, client_id, model, fingerprint):
        self.send = send
        self.client_id = client_id
        self.model = model
        self.fingerprint = fingerprint
        self.identity = send({"operation": "identity"})["identity"]
        check_isolation(client_id, self.identity, model, fingerprint)
        self.dimension = self.identity["dimension"]

    def encode(self, texts, role):
        started = time.perf_counter()
        reply = self.send({"operation": "encode", "expected_model": self.model,
                           "texts": list(texts), "role": role})
        elapsed = time.perf_counter() - started
        check_isolation(self.client_id, reply["identity"], self.model, self.fingerprint)
        if reply["identity"]["resource_id"] != self.identity["resource_id"]:
            raise ValueError("Serving resource changed during measurement")
        values = decode_response(reply, len(texts), self.dimension)
        return values, {"client_round_trip_seconds": elapsed, **reply["telemetry"]}


def private_http_transport(url, *, deadline=60):
    import http.client
    import ipaddress
    import json
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != "http" or not ipaddress.ip_address(parsed.hostname).is_private:
        raise ValueError("EC2 serving must use an explicit private IP")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=deadline)

    def send(value):
        # No transparent application retries. Every failed attempt is observable.
        connection.request("POST", parsed.path, body=json.dumps(value),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"Server request failed: {response.status} {raw[:500]!r}")
        return json.loads(raw)
    return send


def sagemaker_transport(client, endpoint):
    import json

    def send(value):
        response = client.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                                         Accept="application/json", Body=json.dumps(value))
        try:
            return json.loads(response["Body"].read())
        finally:
            response["Body"].close()
    return send
