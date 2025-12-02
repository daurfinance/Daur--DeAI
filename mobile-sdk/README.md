# DAUR Mobile SDK

React Native SDK for DAUR decentralized AI network.

## Features

- ✅ Device capability detection (CPU, GPU, RAM, battery)
- ✅ AI task execution (inference & federated learning)
- ✅ TensorFlow Lite integration
- ✅ Proof-of-Computation generation
- ✅ Reward tracking and management
- ✅ WebSocket connection to orchestrator
- ✅ Background task execution
- ✅ Cryptographic key management

## Installation

```bash
npm install @daur/mobile-sdk
# or
yarn add @daur/mobile-sdk
```

## Quick Start

```typescript
import { DaurClient } from '@daur/mobile-sdk';

// Initialize client
const client = new DaurClient({
  apiUrl: 'https://api.daur-ai.com',
  wsUrl: 'wss://ws.daur-ai.com',
});

await client.initialize();

// Start node
await client.start();

// Listen to events
client.on('taskCompleted', (result) => {
  console.log('Task completed:', result);
});

client.on('rewardEarned', (amount) => {
  console.log('Earned:', amount, 'DAUR tokens');
});

// Get rewards
const rewards = await client.getRewards();
console.log('Total earned:', rewards.totalEarned);
```

## API Reference

### DaurClient

Main client class for interacting with DAUR network.

#### Methods

- `initialize()`: Initialize the client
- `start()`: Start the node
- `stop()`: Stop the node
- `getCapabilities()`: Get device capabilities
- `getRewards()`: Get reward information
- `withdrawRewards(amount, address)`: Withdraw rewards

#### Events

- `connected`: Connected to network
- `disconnected`: Disconnected from network
- `taskReceived`: New task received
- `taskCompleted`: Task completed successfully
- `taskFailed`: Task execution failed
- `rewardEarned`: Reward earned

## Requirements

- React Native >= 0.72.0
- iOS >= 13.0
- Android >= API 24 (7.0)

## License

MIT
