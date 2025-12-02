/**
 * Proof Generator
 * Generates cryptographic proofs of computation
 */

import { createHash } from 'crypto';
import { sign } from 'tweetnacl';
import { encodeBase64 } from 'tweetnacl-util';

export interface ComputationProof {
  hash: string;
  signature: string;
  timestamp: number;
  nonce: string;
}

export class ProofGenerator {
  private keyPair: sign.KeyPair | null = null;

  async initialize(secretKey?: Uint8Array): Promise<void> {
    if (secretKey) {
      this.keyPair = sign.keyPair.fromSecretKey(secretKey);
    } else {
      this.keyPair = sign.keyPair();
    }
  }

  async generate(result: any): Promise<ComputationProof> {
    if (!this.keyPair) {
      throw new Error('ProofGenerator not initialized');
    }

    const timestamp = Date.now();
    const nonce = this.generateNonce();
    
    // Create hash of result + timestamp + nonce
    const data = JSON.stringify({ result, timestamp, nonce });
    const hash = createHash('sha256').update(data).digest('hex');
    
    // Sign the hash
    const signature = sign.detached(
      Buffer.from(hash, 'hex'),
      this.keyPair.secretKey
    );

    return {
      hash,
      signature: encodeBase64(signature),
      timestamp,
      nonce,
    };
  }

  getPublicKey(): string {
    if (!this.keyPair) {
      throw new Error('ProofGenerator not initialized');
    }
    return encodeBase64(this.keyPair.publicKey);
  }

  private generateNonce(): string {
    return Math.random().toString(36).substring(2, 15) +
           Math.random().toString(36).substring(2, 15);
  }
}
