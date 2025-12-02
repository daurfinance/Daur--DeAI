"""
Proof-of-Computation Verifier

Cryptographic verification system for mobile AI computations.
Uses SHA-256 hashing and Ed25519 signatures to prove work was done correctly.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
import nacl.exceptions


@dataclass
class ComputationProof:
    """Proof of computation completion"""
    task_id: str
    node_id: str
    result_hash: str
    timestamp: int
    nonce: str
    signature: str
    public_key: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputationProof':
        return cls(**data)


class ProofGenerator:
    """Generates cryptographic proofs of computation"""
    
    def __init__(self, signing_key: Optional[SigningKey] = None):
        if signing_key:
            self.signing_key = signing_key
        else:
            self.signing_key = SigningKey.generate()
        
        self.verify_key = self.signing_key.verify_key
    
    def generate_proof(
        self,
        task_id: str,
        node_id: str,
        result: Any,
        nonce: Optional[str] = None,
    ) -> ComputationProof:
        """
        Generate a proof of computation.
        
        Args:
            task_id: Unique task identifier
            node_id: Node that performed computation
            result: Computation result
            nonce: Optional nonce for additional randomness
            
        Returns:
            ComputationProof object
        """
        # Generate nonce if not provided
        if nonce is None:
            nonce = self._generate_nonce()
        
        # Create result hash
        result_hash = self._hash_result(result, nonce)
        
        # Create timestamp
        timestamp = int(time.time())
        
        # Create message to sign
        message = self._create_message(task_id, node_id, result_hash, timestamp, nonce)
        
        # Sign message
        signature = self.signing_key.sign(message.encode(), encoder=HexEncoder).signature.decode()
        
        # Create proof
        proof = ComputationProof(
            task_id=task_id,
            node_id=node_id,
            result_hash=result_hash,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            public_key=self.verify_key.encode(encoder=HexEncoder).decode(),
        )
        
        return proof
    
    def _hash_result(self, result: Any, nonce: str) -> str:
        """Hash the computation result with nonce"""
        result_str = json.dumps(result, sort_keys=True)
        data = f"{result_str}:{nonce}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _create_message(
        self,
        task_id: str,
        node_id: str,
        result_hash: str,
        timestamp: int,
        nonce: str,
    ) -> str:
        """Create message to be signed"""
        return f"{task_id}:{node_id}:{result_hash}:{timestamp}:{nonce}"
    
    def _generate_nonce(self) -> str:
        """Generate random nonce"""
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def get_public_key(self) -> str:
        """Get public key for verification"""
        return self.verify_key.encode(encoder=HexEncoder).decode()
    
    def export_private_key(self) -> str:
        """Export private key (keep secure!)"""
        return self.signing_key.encode(encoder=HexEncoder).decode()


class ProofVerifier:
    """Verifies cryptographic proofs of computation"""
    
    def __init__(self):
        self.verified_proofs: Dict[str, ComputationProof] = {}
    
    def verify_proof(
        self,
        proof: ComputationProof,
        result: Any,
        max_age_seconds: int = 3600,
    ) -> Tuple[bool, str]:
        """
        Verify a computation proof.
        
        Args:
            proof: The proof to verify
            result: The actual computation result
            max_age_seconds: Maximum age of proof in seconds
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check timestamp
        current_time = int(time.time())
        age = current_time - proof.timestamp
        
        if age > max_age_seconds:
            return False, f"Proof too old: {age}s > {max_age_seconds}s"
        
        if proof.timestamp > current_time + 60:  # Allow 60s clock skew
            return False, "Proof timestamp in future"
        
        # Verify result hash
        expected_hash = self._hash_result(result, proof.nonce)
        if expected_hash != proof.result_hash:
            return False, "Result hash mismatch"
        
        # Verify signature
        try:
            verify_key = VerifyKey(proof.public_key, encoder=HexEncoder)
            message = self._create_message(
                proof.task_id,
                proof.node_id,
                proof.result_hash,
                proof.timestamp,
                proof.nonce,
            )
            
            verify_key.verify(message.encode(), bytes.fromhex(proof.signature))
        except nacl.exceptions.BadSignatureError:
            return False, "Invalid signature"
        except Exception as e:
            return False, f"Verification error: {str(e)}"
        
        # Store verified proof
        self.verified_proofs[proof.task_id] = proof
        
        return True, "Proof valid"
    
    def _hash_result(self, result: Any, nonce: str) -> str:
        """Hash the computation result with nonce"""
        result_str = json.dumps(result, sort_keys=True)
        data = f"{result_str}:{nonce}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _create_message(
        self,
        task_id: str,
        node_id: str,
        result_hash: str,
        timestamp: int,
        nonce: str,
    ) -> str:
        """Create message that was signed"""
        return f"{task_id}:{node_id}:{result_hash}:{timestamp}:{nonce}"
    
    def get_verified_proof(self, task_id: str) -> Optional[ComputationProof]:
        """Get a previously verified proof"""
        return self.verified_proofs.get(task_id)
    
    def is_verified(self, task_id: str) -> bool:
        """Check if a task has been verified"""
        return task_id in self.verified_proofs


class ProofChain:
    """Chain of proofs for audit trail"""
    
    def __init__(self):
        self.chain: list[ComputationProof] = []
        self.chain_hash: str = "0" * 64
    
    def add_proof(self, proof: ComputationProof) -> None:
        """Add a proof to the chain"""
        # Create chain hash including previous hash
        data = f"{self.chain_hash}:{proof.task_id}:{proof.result_hash}:{proof.signature}"
        self.chain_hash = hashlib.sha256(data.encode()).hexdigest()
        
        self.chain.append(proof)
    
    def verify_chain(self, verifier: ProofVerifier) -> Tuple[bool, str]:
        """Verify the entire proof chain"""
        if not self.chain:
            return True, "Empty chain"
        
        # Recompute chain hash
        computed_hash = "0" * 64
        
        for i, proof in enumerate(self.chain):
            # Verify individual proof (without result, just signature)
            try:
                verify_key = VerifyKey(proof.public_key, encoder=HexEncoder)
                message = f"{proof.task_id}:{proof.node_id}:{proof.result_hash}:{proof.timestamp}:{proof.nonce}"
                verify_key.verify(message.encode(), bytes.fromhex(proof.signature))
            except Exception as e:
                return False, f"Proof {i} invalid: {str(e)}"
            
            # Update chain hash
            data = f"{computed_hash}:{proof.task_id}:{proof.result_hash}:{proof.signature}"
            computed_hash = hashlib.sha256(data.encode()).hexdigest()
        
        if computed_hash != self.chain_hash:
            return False, "Chain hash mismatch"
        
        return True, "Chain valid"
    
    def get_chain_hash(self) -> str:
        """Get current chain hash"""
        return self.chain_hash
    
    def export_chain(self) -> str:
        """Export chain as JSON"""
        return json.dumps({
            'chain_hash': self.chain_hash,
            'proofs': [proof.to_dict() for proof in self.chain],
        }, indent=2)


# Example usage and testing
if __name__ == '__main__':
    print("=== DAUR Proof-of-Computation System ===\n")
    
    # Create generator and verifier
    generator = ProofGenerator()
    verifier = ProofVerifier()
    
    print(f"Node Public Key: {generator.get_public_key()}\n")
    
    # Simulate computation
    task_id = "task_001"
    node_id = "node_12345"
    result = {
        "model": "llama-3.1-8b",
        "inference_time": 1.23,
        "output": "The answer is 42",
        "tokens": 150,
    }
    
    print(f"Task: {task_id}")
    print(f"Node: {node_id}")
    print(f"Result: {result}\n")
    
    # Generate proof
    print("Generating proof...")
    proof = generator.generate_proof(task_id, node_id, result)
    print(f"✅ Proof generated")
    print(f"   Hash: {proof.result_hash}")
    print(f"   Signature: {proof.signature[:32]}...")
    print(f"   Timestamp: {proof.timestamp}\n")
    
    # Verify proof
    print("Verifying proof...")
    is_valid, message = verifier.verify_proof(proof, result)
    
    if is_valid:
        print(f"✅ {message}\n")
    else:
        print(f"❌ {message}\n")
    
    # Test with tampered result
    print("Testing with tampered result...")
    tampered_result = result.copy()
    tampered_result["output"] = "Tampered!"
    
    is_valid, message = verifier.verify_proof(proof, tampered_result)
    print(f"{'✅' if not is_valid else '❌'} Tampered result rejected: {message}\n")
    
    # Create proof chain
    print("Creating proof chain...")
    chain = ProofChain()
    
    for i in range(3):
        task_id = f"task_{i:03d}"
        result = {"task": i, "result": f"output_{i}"}
        proof = generator.generate_proof(task_id, node_id, result)
        chain.add_proof(proof)
    
    print(f"✅ Chain created with {len(chain.chain)} proofs")
    print(f"   Chain hash: {chain.get_chain_hash()}\n")
    
    # Verify chain
    print("Verifying chain...")
    is_valid, message = chain.verify_chain(verifier)
    print(f"{'✅' if is_valid else '❌'} {message}")
