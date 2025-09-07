"""
zk_utils.py — ZK-доказательства для Daur-AI через snarkjs
"""
import subprocess
import json

# Генерация ZK-доказательства через snarkjs

def generate_zk_proof(input_json_path, circuit_wasm_path, zkey_path, proof_path, public_path):
    """
    Генерирует ZK-доказательство для входных данных.
    """
    cmd = [
        "snarkjs", "groth16", "prove",
        zkey_path, circuit_wasm_path, input_json_path,
        "--proof", proof_path,
        "--public", public_path
    ]
    subprocess.run(cmd, check=True)
    with open(proof_path) as f:
        proof = json.load(f)
    return proof

# Верификация ZK-доказательства

def verify_zk_proof(vkey_path, public_path, proof_path):
    """
    Проверяет ZK-доказательство через snarkjs.
    """
    cmd = [
        "snarkjs", "groth16", "verify",
        vkey_path, public_path, proof_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "OK" in result.stdout

# Пример использования:
# proof = generate_zk_proof("input.json", "circuit.wasm", "circuit_final.zkey", "proof.json", "public.json")
# is_valid = verify_zk_proof("verification_key.json", "public.json", "proof.json")
