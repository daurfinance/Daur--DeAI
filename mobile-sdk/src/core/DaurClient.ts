/**
 * DAUR-AI Client
 * 
 * Main client class for interacting with the DAUR-AI network.
 */

import { SDKConfig, NodeInfo, RewardInfo, Task, TaskResult } from '../types';
import { DEFAULT_CONFIG } from '../config';
import { DeviceManager } from './DeviceManager';
import { TaskExecutor } from './TaskExecutor';
import { RewardTracker } from './RewardTracker';
import { NetworkManager } from '../network/NetworkManager';
import { WalletManager } from '../crypto/WalletManager';
import AsyncStorage from '@react-native-async-storage/async-storage';

export class DaurClient {
  private config: SDKConfig;
  private deviceManager: DeviceManager;
  private taskExecutor: TaskExecutor;
  private rewardTracker: RewardTracker;
  private networkManager: NetworkManager;
  private walletManager: WalletManager;
  private nodeInfo: NodeInfo | null = null;
  private isInitialized: boolean = false;
  private isRunning: boolean = false;

  constructor(config: Partial<SDKConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.deviceManager = new DeviceManager();
    this.taskExecutor = new TaskExecutor(this.deviceManager);
    this.rewardTracker = new RewardTracker();
    this.networkManager = new NetworkManager(this.config);
    this.walletManager = new WalletManager(this.config.tonNetwork);
  }

  /**
   * Initialize the DAUR-AI client
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      throw new Error('Client already initialized');
    }

    try {
      // Initialize device manager
      await this.deviceManager.initialize();
      this.log('info', 'Device manager initialized');

      // Initialize wallet
      await this.walletManager.initialize();
      this.log('info', 'Wallet initialized');

      // Load or create node info
      await this.loadNodeInfo();
      this.log('info', 'Node info loaded');

      // Connect to network
      await this.networkManager.connect();
      this.log('info', 'Connected to network');

      // Register node
      await this.registerNode();
      this.log('info', 'Node registered');

      // Setup event listeners
      this.setupEventListeners();

      this.isInitialized = true;
      this.log('info', 'DAUR-AI client initialized successfully');
    } catch (error) {
      this.log('error', `Failed to initialize client: ${error}`);
      throw error;
    }
  }

  /**
   * Start accepting and executing tasks
   */
  async start(): Promise<void> {
    if (!this.isInitialized) {
      throw new Error('Client not initialized. Call initialize() first.');
    }

    if (this.isRunning) {
      this.log('warn', 'Client already running');
      return;
    }

    try {
      // Start listening for tasks
      await this.networkManager.subscribe('tasks', this.handleNewTask.bind(this));
      
      // Start task executor
      this.taskExecutor.start();
      
      // Update node status
      await this.updateNodeStatus('active');
      
      this.isRunning = true;
      this.log('info', 'Client started - accepting tasks');
    } catch (error) {
      this.log('error', `Failed to start client: ${error}`);
      throw error;
    }
  }

  /**
   * Stop accepting tasks
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    try {
      // Stop task executor
      this.taskExecutor.stop();
      
      // Unsubscribe from tasks
      await this.networkManager.unsubscribe('tasks');
      
      // Update node status
      await this.updateNodeStatus('inactive');
      
      this.isRunning = false;
      this.log('info', 'Client stopped');
    } catch (error) {
      this.log('error', `Failed to stop client: ${error}`);
      throw error;
    }
  }

  /**
   * Shutdown the client
   */
  async shutdown(): Promise<void> {
    if (this.isRunning) {
      await this.stop();
    }

    try {
      // Disconnect from network
      await this.networkManager.disconnect();
      
      // Save state
      await this.saveNodeInfo();
      
      this.isInitialized = false;
      this.log('info', 'Client shutdown complete');
    } catch (error) {
      this.log('error', `Failed to shutdown client: ${error}`);
      throw error;
    }
  }

  /**
   * Get node information
   */
  getNodeInfo(): NodeInfo | null {
    return this.nodeInfo;
  }

  /**
   * Get reward information
   */
  async getRewardInfo(): Promise<RewardInfo> {
    return this.rewardTracker.getRewardInfo();
  }

  /**
   * Withdraw rewards to wallet
   */
  async withdrawRewards(amount: number): Promise<string> {
    try {
      const rewardInfo = await this.rewardTracker.getRewardInfo();
      
      if (amount > rewardInfo.available) {
        throw new Error('Insufficient available balance');
      }

      // Request withdrawal from backend
      const result = await this.networkManager.request('withdraw', {
        nodeId: this.nodeInfo?.id,
        amount,
        walletAddress: this.walletManager.getAddress(),
      });

      // Update reward tracker
      await this.rewardTracker.recordWithdrawal(amount, result.txHash);

      this.log('info', `Withdrawal successful: ${amount} DAUR`);
      return result.txHash;
    } catch (error) {
      this.log('error', `Withdrawal failed: ${error}`);
      throw error;
    }
  }

  /**
   * Get device capabilities
   */
  async getDeviceCapabilities() {
    return this.deviceManager.getCapabilities();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SDKConfig>): void {
    this.config = { ...this.config, ...config };
    this.log('info', 'Configuration updated');
  }

  /**
   * Get current configuration
   */
  getConfig(): SDKConfig {
    return { ...this.config };
  }

  // Private methods

  private async loadNodeInfo(): Promise<void> {
    try {
      const stored = await AsyncStorage.getItem('daur_node_info');
      if (stored) {
        this.nodeInfo = JSON.parse(stored);
      } else {
        // Create new node info
        this.nodeInfo = {
          id: this.generateNodeId(),
          walletAddress: this.walletManager.getAddress(),
          status: 'inactive',
          tasksCompleted: 0,
          successRate: 0,
          reputation: 0,
          joinedAt: Date.now(),
          lastActiveAt: Date.now(),
        };
        await this.saveNodeInfo();
      }
    } catch (error) {
      this.log('error', `Failed to load node info: ${error}`);
      throw error;
    }
  }

  private async saveNodeInfo(): Promise<void> {
    if (this.nodeInfo) {
      await AsyncStorage.setItem('daur_node_info', JSON.stringify(this.nodeInfo));
    }
  }

  private async registerNode(): Promise<void> {
    try {
      const capabilities = await this.deviceManager.getCapabilities();
      
      const result = await this.networkManager.request('register_node', {
        nodeId: this.nodeInfo?.id,
        walletAddress: this.walletManager.getAddress(),
        capabilities,
      });

      if (this.nodeInfo) {
        this.nodeInfo.id = result.nodeId;
        await this.saveNodeInfo();
      }

      this.log('info', `Node registered: ${result.nodeId}`);
    } catch (error) {
      this.log('error', `Failed to register node: ${error}`);
      throw error;
    }
  }

  private async updateNodeStatus(status: 'active' | 'inactive'): Promise<void> {
    if (!this.nodeInfo) return;

    this.nodeInfo.status = status;
    this.nodeInfo.lastActiveAt = Date.now();
    await this.saveNodeInfo();

    await this.networkManager.request('update_status', {
      nodeId: this.nodeInfo.id,
      status,
    });
  }

  private async handleNewTask(task: Task): Promise<void> {
    try {
      // Check if device meets requirements
      const canExecute = await this.deviceManager.meetsRequirements(task.requirements);
      
      if (!canExecute) {
        this.log('info', `Task ${task.id} rejected - requirements not met`);
        return;
      }

      // Check battery and network constraints
      if (!this.checkConstraints()) {
        this.log('info', `Task ${task.id} rejected - constraints not met`);
        return;
      }

      // Add task to executor queue
      this.taskExecutor.addTask(task);
      
      this.log('info', `Task ${task.id} accepted`);
    } catch (error) {
      this.log('error', `Failed to handle new task: ${error}`);
    }
  }

  private checkConstraints(): boolean {
    const capabilities = this.deviceManager.getCurrentCapabilities();
    
    if (capabilities.batteryLevel < this.config.minBatteryLevel) {
      return false;
    }

    if (this.config.onlyWhenCharging && !capabilities.isCharging) {
      return false;
    }

    if (this.config.onlyOnWiFi && capabilities.networkType !== 'wifi') {
      return false;
    }

    return true;
  }

  private setupEventListeners(): void {
    // Listen for task completion
    this.taskExecutor.on('taskCompleted', async (result: TaskResult) => {
      await this.handleTaskCompleted(result);
    });

    // Listen for task failure
    this.taskExecutor.on('taskFailed', async (result: TaskResult) => {
      await this.handleTaskFailed(result);
    });
  }

  private async handleTaskCompleted(result: TaskResult): Promise<void> {
    try {
      // Submit result to backend
      const response = await this.networkManager.request('submit_result', {
        nodeId: this.nodeInfo?.id,
        taskId: result.taskId,
        result: result.result,
        proof: result.proof,
        executionTime: result.executionTime,
        resourceUsage: result.resourceUsage,
      });

      // Update rewards
      if (response.reward) {
        await this.rewardTracker.addReward(response.reward, result.taskId, response.txHash);
      }

      // Update node info
      if (this.nodeInfo) {
        this.nodeInfo.tasksCompleted++;
        this.nodeInfo.successRate = response.successRate;
        this.nodeInfo.reputation = response.reputation;
        await this.saveNodeInfo();
      }

      this.log('info', `Task ${result.taskId} completed - reward: ${response.reward} DAUR`);
    } catch (error) {
      this.log('error', `Failed to handle task completion: ${error}`);
    }
  }

  private async handleTaskFailed(result: TaskResult): Promise<void> {
    try {
      // Report failure to backend
      await this.networkManager.request('report_failure', {
        nodeId: this.nodeInfo?.id,
        taskId: result.taskId,
        error: result.error,
      });

      this.log('warn', `Task ${result.taskId} failed: ${result.error}`);
    } catch (error) {
      this.log('error', `Failed to handle task failure: ${error}`);
    }
  }

  private generateNodeId(): string {
    return `node_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  private log(level: string, message: string): void {
    if (!this.config.enableLogging) return;

    const levels = ['debug', 'info', 'warn', 'error'];
    const configLevel = levels.indexOf(this.config.logLevel);
    const messageLevel = levels.indexOf(level);

    if (messageLevel >= configLevel) {
      console.log(`[DAUR-AI] [${level.toUpperCase()}] ${message}`);
    }
  }
}
