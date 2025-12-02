# DAUR Proof-of-Computation System

Cryptographic verification system for mobile AI computations.

## Features

- **SHA-256 Hashing**: Secure hashing of computation results
- **Ed25519 Signatures**: Fast and secure digital signatures
- **Timestamp Verification**: Prevents replay attacks
- **Nonce System**: Additional randomness for security
- **Proof Chains**: Audit trail of computations
- **Result Verification**: Ensures computation integrity

## Usage

```python
from proof_of_computation import ProofGenerator, ProofVerifier

# Create generator (on mobile device)
generator = ProofGenerator()

# Generate proof of computation
proof = generator.generate_proof(
    task_id="task_001",
    node_id="node_12345",
    result={"output": "computation result"},
)

# Verify proof (on backend)
verifier = ProofVerifier()
is_valid, message = verifier.verify_proof(proof, result)

if is_valid:
    print("✅ Computation verified!")
else:
    print(f"❌ Verification failed: {message}")
```

## Security

- Uses NaCl (libsodium) for cryptography
- Ed25519 provides 128-bit security level
- SHA-256 for collision-resistant hashing
- Timestamp validation prevents replay attacks
- Nonce system adds entropy

## License

MIT
