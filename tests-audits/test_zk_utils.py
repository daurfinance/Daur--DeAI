"""
test_zk_utils.py — Тесты для ZK-доказательств Daur-AI
"""
import pytest
from ai_core.zk_utils import generate_zk_proof, verify_zk_proof
import tempfile
import json

@pytest.mark.skip(reason="Requires snarkjs and circuit files")
def test_generate_zk_proof():
    # Моковые файлы для теста
    with tempfile.NamedTemporaryFile("w", delete=False) as input_json, \
         tempfile.NamedTemporaryFile("w", delete=False) as circuit_wasm, \
         tempfile.NamedTemporaryFile("w", delete=False) as zkey, \
         tempfile.NamedTemporaryFile("w", delete=False) as proof, \
         tempfile.NamedTemporaryFile("w", delete=False) as public:
        input_json.write(json.dumps({"a": 1}))
        input_json.flush()
        # В реальном тесте используйте настоящие circuit и zkey
        result = generate_zk_proof(input_json.name, circuit_wasm.name, zkey.name, proof.name, public.name)
        assert isinstance(result, dict)

@pytest.mark.skip(reason="Requires snarkjs and verification key")
def test_verify_zk_proof():
    with tempfile.NamedTemporaryFile("w", delete=False) as vkey, \
         tempfile.NamedTemporaryFile("w", delete=False) as public, \
         tempfile.NamedTemporaryFile("w", delete=False) as proof:
        vkey.write(json.dumps({"key": "value"}))
        vkey.flush()
        result = verify_zk_proof(vkey.name, public.name, proof.name)
        assert isinstance(result, bool)
