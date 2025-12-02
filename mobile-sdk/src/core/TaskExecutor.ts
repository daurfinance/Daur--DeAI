/**
 * Task Executor
 * Executes AI tasks on mobile device
 */

import { Task, TaskResult, TaskStatus, ResourceUsage } from '../types';
import { DeviceManager } from './DeviceManager';
import { AIModelManager } from '../ai/AIModelManager';
import { ProofGenerator } from '../crypto/ProofGenerator';
import { EventEmitter } from 'events';

export class TaskExecutor extends EventEmitter {
  private taskQueue: Task[] = [];
  private runningTasks: Map<string, Task> = new Map();
  private isRunning: boolean = false;
  private aiModelManager: AIModelManager;
  private proofGenerator: ProofGenerator;

  constructor(private deviceManager: DeviceManager) {
    super();
    this.aiModelManager = new AIModelManager();
    this.proofGenerator = new ProofGenerator();
  }

  start(): void {
    this.isRunning = true;
    this.processQueue();
  }

  stop(): void {
    this.isRunning = false;
  }

  addTask(task: Task): void {
    this.taskQueue.push(task);
    this.taskQueue.sort((a, b) => b.priority - a.priority);
    if (this.isRunning) {
      this.processQueue();
    }
  }

  private async processQueue(): Promise<void> {
    while (this.isRunning && this.taskQueue.length > 0) {
      const task = this.taskQueue.shift();
      if (task) {
        await this.executeTask(task);
      }
    }
  }

  private async executeTask(task: Task): Promise<void> {
    const startTime = Date.now();
    this.runningTasks.set(task.id, task);

    try {
      const result = await this.runTask(task);
      const executionTime = Date.now() - startTime;
      
      const taskResult: TaskResult = {
        taskId: task.id,
        status: TaskStatus.COMPLETED,
        result,
        executionTime,
        proof: await this.proofGenerator.generate(result),
        resourceUsage: await this.measureResourceUsage(),
      };

      this.emit('taskCompleted', taskResult);
    } catch (error) {
      const taskResult: TaskResult = {
        taskId: task.id,
        status: TaskStatus.FAILED,
        error: error instanceof Error ? error.message : String(error),
        executionTime: Date.now() - startTime,
        resourceUsage: await this.measureResourceUsage(),
      };

      this.emit('taskFailed', taskResult);
    } finally {
      this.runningTasks.delete(task.id);
    }
  }

  private async runTask(task: Task): Promise<any> {
    switch (task.type) {
      case 'inference':
        return await this.aiModelManager.runInference(task.data);
      case 'training':
        return await this.aiModelManager.runTraining(task.data);
      default:
        throw new Error(`Unsupported task type: ${task.type}`);
    }
  }

  private async measureResourceUsage(): Promise<ResourceUsage> {
    const caps = this.deviceManager.getCurrentCapabilities();
    return {
      cpuUsage: 50, // TODO: Implement actual measurement
      memoryUsage: 512,
      batteryConsumed: 100 - caps.batteryLevel,
      networkDataUsed: 10,
    };
  }
}
