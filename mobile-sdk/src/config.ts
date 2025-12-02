/**
 * SDK Configuration
 */

import { SDKConfig } from './types';

export const DEFAULT_CONFIG: SDKConfig = {
  apiURL: 'https://api.daur-ai.com',
  wsURL: 'wss://ws.daur-ai.com',
  tonNetwork: 'testnet',
  enableBackground: true,
  minBatteryLevel: 20,
  onlyWhenCharging: false,
  onlyOnWiFi: true,
  maxConcurrentTasks: 2,
  enableLogging: true,
  logLevel: 'info',
};

export const TESTNET_CONFIG: SDKConfig = {
  ...DEFAULT_CONFIG,
  apiURL: 'https://testnet-api.daur-ai.com',
  wsURL: 'wss://testnet-ws.daur-ai.com',
  tonNetwork: 'testnet',
};

export const MAINNET_CONFIG: SDKConfig = {
  ...DEFAULT_CONFIG,
  apiURL: 'https://api.daur-ai.com',
  wsURL: 'wss://ws.daur-ai.com',
  tonNetwork: 'mainnet',
};
