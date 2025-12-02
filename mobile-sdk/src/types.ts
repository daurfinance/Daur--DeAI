/**
 * Type definitions for DAUR-AI Mobile SDK
 */

export interface DeviceCapabilities {
  /** Device CPU cores */
  cpuCores: number;
  /** CPU frequency in MHz */
  cpuFrequency: number;
  /** Total RAM in MB */
  totalRAM: number;
  /** Available RAM in MB */
  availableRAM: number;
  /** GPU model */
  gpuModel?: string;
  /** GPU compute capability */
  gpuCapability?: number;
  /** Battery level (0-100) */
  batteryLevel: number;
  /** Is device charging */
  isCharging: boolean;
  /** Device temperature in Celsius */
  temperature?: number;
  /** Network type (wifi, cellular, etc) */
  networkType: string;
  /** Network speed in Mbps */
  networkSpeed?: number;
}

export interface Task {
  /** Unique task ID */
  id: string;
  /** Task type */
  type: TaskType;
  /** Task priority (1-10) */
  priority: number;
  /** Task data/payload */
  data: any;
  /** Required device capabilities */
  requirements: TaskRequirements;
  /** Reward amount in DAUR tokens */
  reward: number;
  /** Task deadline timestamp */
  deadline: number;
  /** Task metadata */
  metadata?: Record<string, any>;
}

export enum TaskType {
  INFERENCE = 'inference',
  TRAINING = 'training',
  FEDERATED_LEARNING = 'federated_learning',
  DATA_PROCESSING = 'data_processing',
  MODEL_EVALUATION = 'model_evaluation',
  REVERSE_AI = 'reverse_ai',
}

export interface TaskRequirements {
  /** Minimum CPU cores */
  minCPUCores?: number;
  /** Minimum RAM in MB */
  minRAM?: number;
  /** Requires GPU */
  requiresGPU?: boolean;
  /** Minimum battery level */
  minBattery?: number;
  /** Requires charging */
  requiresCharging?: boolean;
  /** Requires WiFi */
  requiresWiFi?: boolean;
  /** Maximum execution time in seconds */
  maxExecutionTime?: number;
}

export interface TaskResult {
  /** Task ID */
  taskId: string;
  /** Execution status */
  status: TaskStatus;
  /** Result data */
  result?: any;
  /** Error message if failed */
  error?: string;
  /** Execution time in milliseconds */
  executionTime: number;
  /** Proof of computation */
  proof?: ComputationProof;
  /** Resource usage */
  resourceUsage: ResourceUsage;
}

export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface ComputationProof {
  /** Proof type */
  type: 'hash' | 'zksnark' | 'signature';
  /** Proof data */
  data: string;
  /** Timestamp */
  timestamp: number;
  /** Device signature */
  signature: string;
}

export interface ResourceUsage {
  /** CPU usage percentage */
  cpuUsage: number;
  /** Memory usage in MB */
  memoryUsage: number;
  /** GPU usage percentage */
  gpuUsage?: number;
  /** Battery consumed percentage */
  batteryConsumed: number;
  /** Network data used in MB */
  networkDataUsed: number;
}

export interface RewardInfo {
  /** Total earned rewards */
  totalEarned: number;
  /** Available balance */
  available: number;
  /** Pending rewards */
  pending: number;
  /** Withdrawn amount */
  withdrawn: number;
  /** Reward history */
  history: RewardTransaction[];
}

export interface RewardTransaction {
  /** Transaction ID */
  id: string;
  /** Transaction type */
  type: 'earned' | 'withdrawn' | 'bonus';
  /** Amount */
  amount: number;
  /** Timestamp */
  timestamp: number;
  /** Related task ID */
  taskId?: string;
  /** Transaction hash on blockchain */
  txHash?: string;
}

export interface AIModel {
  /** Model ID */
  id: string;
  /** Model name */
  name: string;
  /** Model version */
  version: string;
  /** Model type */
  type: 'tflite' | 'onnx' | 'pytorch';
  /** Model size in MB */
  size: number;
  /** Model URL */
  url: string;
  /** Model hash for verification */
  hash: string;
  /** Required capabilities */
  requirements: TaskRequirements;
  /** Model metadata */
  metadata?: Record<string, any>;
}

export interface SDKConfig {
  /** Backend API URL */
  apiURL: string;
  /** WebSocket URL */
  wsURL: string;
  /** TON blockchain network */
  tonNetwork: 'mainnet' | 'testnet';
  /** Enable background execution */
  enableBackground: boolean;
  /** Minimum battery level to accept tasks */
  minBatteryLevel: number;
  /** Only accept tasks when charging */
  onlyWhenCharging: boolean;
  /** Only accept tasks on WiFi */
  onlyOnWiFi: boolean;
  /** Maximum concurrent tasks */
  maxConcurrentTasks: number;
  /** Enable logging */
  enableLogging: boolean;
  /** Log level */
  logLevel: 'debug' | 'info' | 'warn' | 'error';
}

export interface NodeInfo {
  /** Node ID */
  id: string;
  /** Node wallet address */
  walletAddress: string;
  /** Node status */
  status: 'active' | 'inactive' | 'suspended';
  /** Total tasks completed */
  tasksCompleted: number;
  /** Success rate */
  successRate: number;
  /** Reputation score */
  reputation: number;
  /** Joined timestamp */
  joinedAt: number;
  /** Last active timestamp */
  lastActiveAt: number;
}
