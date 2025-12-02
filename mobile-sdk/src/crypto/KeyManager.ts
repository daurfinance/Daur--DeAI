/**
 * Key Manager
 * Manages cryptographic keys for the node
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { sign } from 'tweetnacl';
import { encodeBase64, decodeBase64 } from 'tweetnacl-util';

export class KeyManager {
  private keyPair: sign.KeyPair | null = null;

  async initialize(): Promise<void> {
    const stored = await AsyncStorage.getItem('daur_keypair');
    
    if (stored) {
      const { publicKey, secretKey } = JSON.parse(stored);
      this.keyPair = {
        publicKey: decodeBase64(publicKey),
        secretKey: decodeBase64(secretKey),
      };
    } else {
      this.keyPair = sign.keyPair();
      await this.save();
    }
  }

  getPublicKey(): string {
    if (!this.keyPair) {
      throw new Error('KeyManager not initialized');
    }
    return encodeBase64(this.keyPair.publicKey);
  }

  getSecretKey(): Uint8Array {
    if (!this.keyPair) {
      throw new Error('KeyManager not initialized');
    }
    return this.keyPair.secretKey;
  }

  async regenerate(): Promise<void> {
    this.keyPair = sign.keyPair();
    await this.save();
  }

  private async save(): Promise<void> {
    if (!this.keyPair) {
      throw new Error('No keypair to save');
    }

    const data = {
      publicKey: encodeBase64(this.keyPair.publicKey),
      secretKey: encodeBase64(this.keyPair.secretKey),
    };

    await AsyncStorage.setItem('daur_keypair', JSON.stringify(data));
  }
}
